# Matcher tuning on ETH-80 — measured surface

Pre-registration: `work/topology/TUNING_PREREGISTRATION.md` (written before any accuracy number).

## Data and graphs

Instance-disjoint split: 24 references, 48 dev queries, 48 held-out queries (8 classes; three reference instances per class). Graph budget 512 edges; mean 321.3 nodes / 144.3 edges over 120 graphs.

## Dev surface (60 alignment x 216 score cells = 12 960, plus reference rows)

Two score formulas, from cached facts:

```
pooled:     S = 2 * (node_credit + w*edge_credit) / (na+nb+edge_mass)
normalized: node_term = node_credit/(na+nb)
            edge_term = edge_credit/(checkable_a+checkable_b)
            node_cov = 2*matched/(na+nb)
            edge_cov = preserved/checkable
            S = 2*((1-w)*node_term*node_cov**delta + w*edge_term*edge_cov**gamma)
```

Best dev cell (offline score-time formula): beta=0.75, unmatched=0.15, occlusion=0.5, formula=pooled, w=0.1, gamma=0.5, delta=0.0, denom=checkable, isolated=downweight -> top-1 0.5417, AUC 0.8155.

Node-only reference (best alignment for that row): top-1 0.4167 (95% interval 0.2708–0.5625), AUC 0.7321 (95% interval 0.6458–0.8155; beta=0.3, unmatched=derived, occlusion=0.0).

The strongest node-only variant (`norm_node_only_d0`, normalized, w=0, delta 0) reaches top-1 0.5625 / AUC 0.7917, so the winner is +0.0238 AUC over a pure node score.

Winner dev top-1 95% interval 0.4167–0.6667; AUC 95% interval by query frame 0.7411–0.8869 (bootstrap mean 0.8163).

### Production transfer (post-hoc check)

One dev pair (`apple1-090-000` vs `apple4-035-135`), winner alignment and score settings, run through production `joint_mcs` and the offline scorer:

| isolated | prod map | offline map | same map | prod score | offline w=0.1 | offline w=1.0 |
|---|---|---|---|---|---|---|
| keep | 230 | 230 | yes | 0.867295 | 0.669147 | 0.867295 |
| downweight | 229 | 230 | no | 0.600530 | 0.418720 | 0.616868 |

Production's pooled branch never reads `MATCH_EDGE_WEIGHT`: it always gives the edge term full weight. At isolated=keep the two implementations return the same mapping and the same node, edge and denominator components; production's score equals the offline w=1.0 score exactly, while the selected w=0.1 gives 0.669147. At isolated=downweight production applies the weight inside node similarity before alignment (229 vs 230 mapped nodes), so even w=1.0 differs (0.600530 vs 0.616868). The selected cell is therefore a score-time study configuration, not a deployable parameter set: deploying it needs `MATCH_EDGE_WEIGHT` in the pooled branch and a decision on where the isolated weight applies. No cell in the grid is exactly production-equivalent: production's pooled score is the unmeasured w=1 edge, and its separate-term branch has no coverage exponents.

### Top 12 cells by dev AUC

| beta | unmatched | occlusion | formula | w | gamma | delta | denom | isolated | top-1 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 0.15 | 0.5 | pooled | 0.1 | 0.5 | 0.0 | checkable | downweight | 0.5417 | 0.8155 |
| 0.75 | 0.15 | 0.5 | pooled | 0.1 | 0.5 | 0.0 | per_side | downweight | 0.5417 | 0.8125 |
| 0.5 | 0.3 | 0.5 | pooled | 0.5 | 0.5 | 0.0 | all | keep | 0.5208 | 0.8125 |
| 0.5 | 0.3 | 0.5 | pooled | 0.5 | 0.0 | 0.0 | all | keep | 0.5000 | 0.8125 |
| 0.5 | 0.3 | 0.5 | pooled | 0.5 | 0.0 | 0.0 | checkable | keep | 0.5000 | 0.8125 |
| 0.5 | 0.3 | 0.5 | pooled | 0.5 | 0.0 | 0.0 | per_side | keep | 0.5000 | 0.8125 |
| 0.75 | 0.15 | 0.25 | pooled | 0.1 | 0.5 | 0.0 | checkable | downweight | 0.5625 | 0.8095 |
| 0.75 | 0.15 | 0.25 | pooled | 0.1 | 0.5 | 0.0 | per_side | downweight | 0.5625 | 0.8095 |
| 0.5 | 0.3 | 0.25 | pooled | 0.5 | 0.5 | 0.0 | all | downweight | 0.5417 | 0.8095 |
| 0.75 | 0.0 | 0.25 | pooled | 0.1 | 0.5 | 0.0 | checkable | downweight | 0.5417 | 0.8095 |
| 0.75 | 0.0 | 0.25 | pooled | 0.1 | 0.5 | 0.0 | per_side | downweight | 0.5417 | 0.8095 |
| 0.75 | 0.15 | 0.5 | pooled | 0.1 | 0.5 | 0.0 | all | downweight | 0.5417 | 0.8095 |

Every cell is in `results/tune/dev_surface.csv` (12960 tuned cells; 240 reference rows).

### Does the edge weight matter?

The pooled formula cannot make edges decisive: on a 96-pair dev sample (4 queries x 24 references) the node:edge credit ratio has median 5.05 (mean 6.10, range 1.65-25.76), so even w=0.75 leaves the edge term a small part of the pooled numerator (the pre-registration's ~10x estimate was high). The normalized formula makes w a true blend, so it can. Best AUC per w, marginalising the other knobs:


**pooled**

| w | cells | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|---|
| 0.1 | 1620 | 0.8155 | 0.7236 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |
| 0.25 | 1620 | 0.8065 | 0.7269 | 0.5208 | pooled_w0.25_g0.0_all_downweight |
| 0.5 | 1620 | 0.8125 | 0.7295 | 0.5208 | pooled_w0.50_g0.5_all_keep |
| 0.75 | 1620 | 0.8036 | 0.7261 | 0.5000 | pooled_w0.75_g0.0_all_downweight |

**normalized**

| w | cells | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|---|
| 0.1 | 1620 | 0.8095 | 0.7203 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| 0.25 | 1620 | 0.7976 | 0.6942 | 0.5625 | norm_w0.25_g1.0_d0.0_downweight |
| 0.5 | 1620 | 0.7887 | 0.6537 | 0.5417 | norm_w0.50_g1.0_d0.0_downweight |
| 0.75 | 1620 | 0.7649 | 0.6254 | 0.5208 | norm_w0.75_g1.0_d1.0_keep |

At one fixed alignment cell (beta 0.75, unmatched 0.15, occlusion 0.5), so only the score changes:

| row | top-1 | AUC |
|---|---|---|
| pooled node-only | 0.4375 | 0.7202 |
| normalized node-only, delta 0 | 0.5208 | 0.7887 |
| normalized w=0.10, gamma 0, delta 0 | 0.4375 | 0.7798 |
| normalized w=0.75, gamma 1, delta 1 | 0.1458 | 0.6607 |

In the normalized arm, edge credit does not raise AUC over the node-only row at that mapping (w=0.10 is slightly below it), and at w=0.75, where the edge term is three quarters of the blend, AUC falls to 0.6607. Across all mappings the normalized arm's best AUC falls monotonically with w. The pooled winner's own AUC does not isolate edge credit: its w=0 reference at the same mapping scores 0.7202 with a different denominator, while the normalized w=0 row, which differs only in the edge blend, scores 0.7887 and does not improve when edges enter. This is the measured answer to the question the pooled formula could not ask: in this measurement, the edges are not additive class evidence at this weight range, and making them decisive hurts.

### Reference rows (same alignment grid, not tuned)

Best alignment cell for each reference row (all 60 alignment cells per row are in the CSV):

| name | beta | unmatched | occlusion | top-1 | AUC |
|---|---|---|---|---|---|
| node_only | 0.3 | derived | 0.0 | 0.4167 | 0.7321 |
| norm_node_only_d0 | 0.3 | 0.0 | 0.5 | 0.5625 | 0.7917 |
| norm_node_only_d1 | 0.75 | 0.15 | 0.5 | 0.5000 | 0.7768 |
| presence_excluded | 0.5 | 0.3 | 0.5 | 0.5208 | 0.7679 |

### Marginals (all tuned cells)

**beta**

| beta | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| 0.0 | 0.8095 | 0.7357 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| 0.15 | 0.8006 | 0.7086 | 0.5417 | pooled_w0.50_g0.0_all_keep |
| 0.3 | 0.7976 | 0.6983 | 0.5417 | pooled_w0.50_g0.5_per_side_keep |
| 0.5 | 0.8125 | 0.6904 | 0.5208 | pooled_w0.50_g0.5_all_keep |
| 0.75 | 0.8155 | 0.6667 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |

**unmatched**

| unmatched | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| 0.0 | 0.8095 | 0.7194 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |
| 0.15 | 0.8155 | 0.7153 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |
| 0.3 | 0.8125 | 0.6895 | 0.5208 | pooled_w0.50_g0.5_all_keep |
| derived | 0.8095 | 0.6756 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |

**occlusion**

| occlusion | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| 0.0 | 0.8095 | 0.7000 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| 0.25 | 0.8095 | 0.6987 | 0.5625 | pooled_w0.10_g0.5_checkable_downweight |
| 0.5 | 0.8155 | 0.7012 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |

**formula**

| formula | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| normalized | 0.8095 | 0.6734 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| pooled | 0.8155 | 0.7265 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |

**isolated**

| isolated | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| downweight | 0.8155 | 0.7299 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |
| exclude | 0.7619 | 0.6405 | 0.5000 | norm_w0.25_g0.0_d1.0_exclude |
| keep | 0.8125 | 0.7294 | 0.5208 | pooled_w0.50_g0.5_all_keep |

**denom (pooled cells only)**

| denom | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| all | 0.8125 | 0.7291 | 0.5208 | pooled_w0.50_g0.5_all_keep |
| checkable | 0.8155 | 0.7272 | 0.5417 | pooled_w0.10_g0.5_checkable_downweight |
| per_side | 0.8125 | 0.7232 | 0.5417 | pooled_w0.10_g0.5_per_side_downweight |

**gamma (normalized cells only)**

| gamma | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| 0.0 | 0.8095 | 0.6669 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| 0.5 | 0.8006 | 0.6730 | 0.4583 | norm_w0.10_g0.5_d0.0_downweight |
| 1.0 | 0.8095 | 0.6802 | 0.4792 | norm_w0.10_g1.0_d0.0_downweight |

**delta (normalized cells only)**

| delta | max AUC | mean AUC | best top-1 | best cell |
|---|---|---|---|---|
| 0.0 | 0.8095 | 0.6695 | 0.5208 | norm_w0.10_g0.0_d0.0_downweight |
| 0.5 | 0.8006 | 0.6739 | 0.4792 | norm_w0.10_g1.0_d0.5_downweight |
| 1.0 | 0.8006 | 0.6767 | 0.5208 | norm_w0.10_g1.0_d1.0_downweight |

## Held-out (winner fixed on dev, no re-selection)

| scoring | top-1 | 95% top-1 interval | AUC | 95% AUC interval |
|---|---|---|---|---|
| best configuration (offline formula) | 0.5208 | 0.3750–0.6667 | 0.7649 | 0.6756–0.8542 |
| node-only | 0.5208 | 0.3750–0.6667 | 0.7679 | 0.6786–0.8571 |
| edges rewired | 0.5208 | 0.3958–0.6667 | 0.7560 | 0.6667–0.8482 |
| no edges | 0.4792 | 0.3333–0.6250 | 0.7411 | 0.6488–0.8333 |

Node-only reuses the winner's real-edge mapping with w=0; the no-edge row re-solves with empty edge sets, so it also changes the correspondence. The rewired control keeps the edge count and degree but drops edge parameters.


Winner correspondence, averaged over every query x reference pair:

| split | pairs | matched nodes | matched edges | edge preservation | graded edge similarity |
|---|---|---|---|---|---|
| dev | 1152 | 237.7 | 55.5 | 0.3824 | 0.5184 |
| held-out | 1152 | 242.6 | 54.8 | 0.3734 | 0.5178 |

Deltas against the selected configuration on the same held-out queries (negative = worse than real edges): node-only AUC +0.0030 / top-1 +0.0000; rewired AUC -0.0089 / top-1 +0.0000; no edges AUC -0.0238 / top-1 -0.0417. The intervals overlap, so this ablation does not separate the edge structure from node-only scoring. The rewired control lands close to the real graph rather than at the no-edge level, which is not the pattern expected if the specific edge structure carried the class signal. Top-1 is also well above the 0.125 chance level, but the held-out point estimate is below the dev estimate, as expected after selecting the dev maximum.


## Known-transform instrument (corrected)

Grid pitch 16.0 px (density 16 on a 256 px image); tolerance 8.0 px = half pitch. Warp: np.roll(image, (dy,dx), axis=(0,1)); positive dx moves content right.

| shift | nodes | with position | truth preserved | truth frac | anchor-cell frac | matcher frac | Jaccard |
|---|---|---|---|---|---|---|---|
| [0, 0] | 872 | 857 | 857/857 | 1.000 | 1.000 | 1.000 | 1.000 |
| [2, 0] | 876 | 861 | 754/857 | 0.880 | 0.904 | 0.792 | 0.598 |
| [-2, 0] | 876 | 861 | 730/857 | 0.852 | 0.880 | 0.780 | 0.609 |
| [3, 0] | 873 | 858 | 716/857 | 0.835 | 0.865 | 0.718 | 0.513 |
| [-3, 0] | 866 | 851 | 715/857 | 0.834 | 0.853 | 0.745 | 0.570 |

At 0 px: truth 1.000, Jaccard 1.000. The historical 403 -> 164 counts are not reproducible from this tree: the current graph has 872 nodes at 0 px and 876 at +2 px, with 857 and 861 positioned nodes, so no 60% node loss exists. A 2 px shift is an eighth of the 16 px pitch: only centroids within 2 px of a cell boundary can cross one, so it cannot produce a 60% loss. The mechanism consistent with both old numbers is that the counted set changed at extraction (margin filter and anchor-cell quantizer), not that detection collapsed; neither old number is reported as a stability result.

## Masked cells

The final surface has **zero masked cells**: all 48 dev queries x 8 classes x 220 score cells are finite (13 200 score rows x 48 queries x 8 classes). The checks are mechanical: the worker seeds its array with -inf (not NaN, because `max(nan, x)` keeps nan and would silently mask a class), refuses to write a query unless every class that has a reference also has a finite score, and `combine` refuses to stack the surface if any non-finite entry remains, writing `masked_cells.json` with the query and class.

The historical sentinels were narrower than a missing class: the npz stores one score per (query, score cell, class), with references aggregated before the save. One file, `apple4-035-135.npz`, had been overwritten by a manual 4-reference probe covering only apple and car. The six classes with no reference in that probe (cow, cup, dog, horse, pear, tomato) kept the `-1e30` initialiser: at most 6 x 6 600 = 39 600 sentinel entries. That is an upper bound reconstructed from the grid shape; the exact count is not recoverable because the probe file was replaced. The full 24-reference worker replaced that file, and no sentinel was found in the other dev files that were checked (the first ten completed files). The old `-1e30` initialiser would have hidden the same condition in the final run; the -inf initialiser plus the two assertions make it impossible to report a surface with a silently dropped cell.

## Determinism

Winner under `PYTHONHASHSEED=0` and `1729`: hashes MATCH.

## Provenance

| file | status | added | removed | churn-only |
|---|---|---|---|---|
| src/matching/joint_mcs.py | new (no baseline) | n/a | n/a | False |
| src/matching/alignment.py | modified | 88 | 21 | False |
| src/parameters/_text.py | modified | 99 | 0 | False |
| tests/matcher/conftest.py | modified | 7 | 36 | False |
| tests/matcher/test_phase3_f1_f4.py | new (no baseline) | n/a | n/a | False |

Baseline: `E:\Take25\work\bak\pre_percept5_20261114_161025` (`work/` is untracked; the tracked repo `src/` is a different copy). CRLF normalized before the line diff.

## Compute

48 dev queries, 60 alignment cells x 220 score cells, 12 workers. Alignment-pair evaluations 69120; score evaluations 15206400. Wall time 3458 s across 2 invocations (12 queries / 1093 s, 36 queries / 2365 s); the second resumed exactly the queries missing from the first, with the same code and grid.

## What was not run

- No second corpus, no scaling beyond the pre-registered split.
- No tuning of graph budget, views, or the front end.
- No per-cell bootstrap intervals on the full surface (the winner and node-only rows carry them).
- The held-out metrics come from two invocations of the same winner: the first was cut off by a tool timeout after 16 of 48 queries, and the second resumed the remaining 32 with no configuration change. An earlier attempt aborted before writing any score because the node-only ablation mode was not routed to the real graph; that routing fix changes no score.
- The 403/164 historical counts are not used as a result; the corrected instrument is reported instead.
- The selected cell is not production-equivalent (Production transfer): the held-out numbers measure the offline score-time formula, not `joint_mcs` as it ships.

