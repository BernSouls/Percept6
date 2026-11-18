# Pre-registration — matcher tuning on ETH-80 (written before accuracy)

This file is written before any tuning accuracy number exists. The only
measurements that informed it are the one-pair solver timing
(`_bench_solver.py`: 39 alignment configs, 11.2 s/pair) and the front-end
cost already on record (25.2 s/graph, 512-edge budget).

## Data (instance-disjoint, three reference instances per class)

ETH-80, 8 classes x 10 instances x 41 views. Ranks are 0-based positions in
each class's sorted instance list (car's numbers are non-contiguous).

| role | ranks | views | images |
|---|---|---|---|
| reference ("training" = stored) | 0, 1, 2 | -090-000 | 24 (3/class) |
| dev queries | 3, 4, 5 | -090-000, -035-135 | 48 (6/class) |
| held-out queries | 6, 7, 8 | -090-000, -035-135 | 48 (6/class) |

Rank 9 is unused. No query instance appears as a reference; dev and held-out
are disjoint by instance. Queries are paired by frame only within a view
(two views of the same instance are separate query frames for the bootstrap).

## Matcher grid (every cell reported)

Alignment parameters (these change the correspondence; each is solved fresh):

* `MATCH_SYSTEMATICITY_WEIGHT` beta in {0.0, 0.15, 0.3, 0.5, 0.75}
* `MATCH_UNMATCHED_COST` in {0.0, 0.15, 0.3, derived}
* occlusion credit (`MATCH_ALIGN_UNMAPPED_CREDIT`) in {0.0, 0.25, 0.5}

Score parameters (re-scored from the cached correspondence, no re-matching):

* edge weight w (`MATCH_EDGE_WEIGHT`) in {0.1, 0.25, 0.5, 0.75}
* coverage exponent gamma (`MATCH_COVERAGE_EXPONENT_GAMMA`) in {0.0, 0.5, 1.0}
* edge denominator in {all, checkable, per_side}
* isolated handling in {keep, downweight 0.5, exclude (weight 0.0)}

The isolated values are applied as SCORE-TIME weights on one common
correspondence: removal from the mapping is not used (the user's instruction
is to replace removal with a down-weight, and paired arms need the same
mapping). `exclude` is the w=0.0 endpoint of that weight, not a different
node set.

Reference rows in every table (never used for selection):
node-only (w = 0, same mapping) and the presence-only-excluded score
(`MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE` on) at w = 0.5, gamma = 1,
denominator all, isolated keep. The hard categorical gate is otherwise OFF
for the whole sweep; categorical credit is graded when it is on.

Total alignment solves: 5 beta x 4 unmatched x 3 occlusion = 60 cells per
direction per pair, with the beta = 0 cells sharing one solve (occlusion
cannot change a seed-only mapping): 52 unique solves per direction.
Total score cells: 60 x 108 = 6 480, plus the two reference rows.
(Amendment 1 below supersedes this with 216 tuned cells and four reference
rows; the original text is kept so the change is visible.)

## Selection rule (fixed before the sweep)

Select the configuration with the highest **dev pair-level AUC**; ties break
on dev top-1, then on grid order. AUC uses every (query, true class,
negative class) pair and is the stronger statistic at n = 48 queries.
Held-out is run ONCE with that configuration, unchanged. The node-only row is
reported beside it as a reference, not a gate.

## Pair-level AUC and intervals

For query q with true class c, every negative class d != c contributes a
pair; the pair is 1 if S(q,c) > S(q,d), 0.5 on a tie, else 0. AUC is the mean
over all pairs. The 95 % interval is a paired bootstrap over QUERY FRAMES
(2 000 resamples with replacement, 2.5/97.5 percentiles).

## Edge ablation (on the held-out winner)

* real: the winner's own edges.
* rewired: each reference's edges degree-preservingly rewired (undirected
  double-edge swaps within edge type; directed destinations permuted within
  type), deterministic seed, edge parameters dropped (topology-only control).
* no-edge: alignment and score with empty edge sets (node-only).

If destroying the edges does not lower the held-out number, that is reported
as the finding; tuning continues rather than stopping.

## Known-transform instrument (fixed before running)

Image `data/ETH-80-master/1/2/apple2-022-000.png` (256 x 256). Content
displacements (dx, dy) in {(0,0), (+2,0), (-2,0), (+3,0), (-3,0)} applied with
`np.roll(image, (dy,dx), axis=(0,1))`; the truth position of a node is
`p + (dx,dy)`. The warp sign is therefore "positive dx moves content right".
Matching is by CONTINUOUS detection centroid with an 8 px tolerance (half the
16 px retinotopic grid: density 16, side/16 = 16 px pitch). The anchor-CELL
instrument (integer cell = floor(coord/16)) is measured alongside, so the
historical collapse can be attributed to the quantizer rather than to
detection instability. 0 px must be 100 % / Jaccard 1.0.

## Determinism

The winner is run under `PYTHONHASHSEED=0` and `PYTHONHASHSEED=1729`; the
per-query score vectors (float hex) and predicted labels are hashed and
compared. A mismatch is reported, not hidden.

## Compute

Graph build: 120 graphs (24 references + 48 dev + 48 held-out) at ~25 s each,
12 workers. Dev sweep: 1 152 pairs x ~17 s / 12 workers, ~30 min. Held-out,
ablation and determinism are winner-only. The measured compute is reported.

## Amendment 1 — before the surface was measured (2026-11-15)

The originally registered pooled formula was `(node_credit + w*edge_credit)/
(na+nb+edge_mass)`.  In that form `node_credit` is roughly 10x
`edge_credit`, so even w = 0.75 adds only a few percent of the score and the
edges cannot be decisive.  That is a formula artifact, so the surface is NOT
used to claim "edges do not matter".

Added, and swept as its own labelled arm (the pooled arm is kept):

    node_term = node_credit / (na + nb)
    edge_term = edge_credit / edge_mass       (edge_mass = checkable_a+checkable_b)
    node_cov  = 2*matched / (na + nb)
    edge_cov  = preserved / checkable
    score     = 2*((1-w)*node_term*node_cov**delta + w*edge_term*edge_cov**gamma)

with delta in {0.0, 0.5, 1.0} added to the grid, w in {0.1, 0.25, 0.5,
0.75}, gamma in {0.0, 0.5, 1.0} and isolated in {keep, downweight 0.5,
exclude (weight 0.0)}.  This adds 108 normalized cells (4 x 3 x 3 x 3), for
216 tuned cells per alignment cell; every cell is reported. The reference
rows become four: `node_only` (pooled, w=0), `norm_node_only_d0` and
`norm_node_only_d1` (normalized, w=0, delta 0/1), and `presence_excluded`
(pooled, w=0.5, presence counts off). None enters selection.

Masked cells are now impossible to report silently. Worker scores are
initialised to -inf, not NaN, because Python's `max(nan, x)` keeps nan and a
NaN initialiser would silently mask a class. A worker refuses to write a
query unless every class that has a reference has a finite score, and
`combine` refuses to build the surface if any non-finite entry remains (and
writes `masked_cells.json` naming the query and class).  The earlier `-1e30` values were the initialisation sentinel, not a
score; one file had them because a 4-reference probe overwrote it, and that
file was replaced by the full 24-reference worker.  The final surface is
expected to have zero masked cells, and this is checked, not assumed.

## Amendment 2 — the selected cell no longer functions (2026-11-15)

Amendment 2 is being written before any new surface is measured, per the
stop rule. The audit that produced it (all on this tree): the joint
correspondence objective in matching/alignment.py is now additive
(q = sim*(1-beta) + beta*rel; an edgeless side carries rel = 0.0 with
MATCH_RELATIONAL_CONJUNCTION off, and no relational conjunction otherwise),
the edgeless-side credit is zero rather than unitary, and an edgeless pair
can therefore never earn relational support. TUNING_REPORT.md's numbers were
produced under the old multiplicative modulation and are stale.

The transfer-check probe (transfer_check.py) of the selected dev cell
(pooled_w0.10_g0.5_checkable_downweight) on apple1-090-000 <-> apple4-035-135
measured the collapse directly: at beta = 0.75 with the isolated weight 0.5,
q(isolated) = 0.25 * 0.5 * signature <= 0.125, and the absolute price
MATCH_UNMATCHED_COST = 0.15. **0 of the 10,054 positive isolated candidates
can ever pay the price**; the 65-76% of nodes with degree 0 on this pair are
unmatchable by construction, and the matched set fell from 229 to 56.

Stated consequence before the re-registered grid exists: **the retune is
mandatory, not optional.** The selected configuration was optimised against
an objective scale that no longer exists. In the re-registered grid the
absolute MATCH_UNMATCHED_COST is replaced by the derived variant (or swept
jointly with beta), per the Step 4 bounded grid:

* beta in {0.15, 0.3, 0.5}
* unmatched in {0.0, derived}
* occlusion in {0.0, 0.5}

expanded only if the bounded surface is flat; AUC remains the selection
statistic; held-out runs ONCE on the winner; node-only row in every table;
determinism on the winner. No number from Amendment 2's grid may be read
against Amendment 1's tables.
