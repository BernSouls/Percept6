# MISSION 20 — data split and the four tasks (registered before any graph exists)

Written 2026-11-18, committed on its own, before a single ETH-80 graph has been built.
Nothing in this file was chosen after seeing a score: no ETH-80 image had been matched, no
FIT number exists yet, and the only measurement in the repository at this point is the
parity check of the ported M19 scorer on the old DEV cells.

## 1. The corpus

`Topology/fetch_corpus.py --fetch` fetched the `chenchkx/ETH-80` mirror (322 MB tarball,
kept in `.cache/`, outside git) and verified it: **8 classes × 10 instances × 41 views =
3,280 images**, one 256×256 object image plus one mask per view, 41 distinct
`(azimuth, elevation)` keys, every key present in all 80 objects. Layout is
`data/ETH-80-master/<class>/<instance>/<name>-<az>-<el>.png` (masks beside the image as
`<name>-<az>-<el>-map.png`; this mirror has no `maps/` subdirectory).

Two defects in the fetcher were found and fixed on the way, both recorded in the file:

* `extractall` sat one indent too deep inside the member loop, so it ran **once per
  member** — 6,560 extractions of the whole archive. Measured: more than an hour of CPU
  and one class of eight on disk before the process was stopped. The download was never
  the problem.
* `verify()` asserted a layout the mirror does not have (a `maps/` subdirectory and two
  hard-coded view keys, one of which — `-035-135` — exists only by accident), so it would
  have rejected a good corpus. It now checks the invariants the work actually needs:
  instances per class, views per instance, one mask per image, a 256×256 image per object,
  and that the view-key *set* is identical across all 80 objects.

Verified file: `Topology/results/corpus_verify.json` (`verified: true`).

## 2. The split

`Topology/placement_m20_splits.py` → `Topology/results/placement/m20_splits.json`.

* **Unit is the OBJECT**, never the view. FIT / DEV / FRESH are disjoint by object, so no
  instance can be a gallery reference in one split and a query in another.
* Per class the ten instances are ordered by `sha256("M20-split-v1-2026-11-18|<class>|<instance>")`
  and dealt **5 / 3 / 2 → FIT / DEV / FRESH**. The salt and the deal are constants in the
  script; `--check` re-derives the file and diffs, which is how determinism is proven
  without a self-referential hash.
* Every class is present in every split. The class set is a parameter
  (`--classes apple,pear,tomato`), and a subset re-deals inside itself with the same rule,
  so a confusable subset still covers all three splits.
* Views are **not** split: an object keeps all 41, because T1 needs "same object, other
  view" and thinning views would shrink the thing under test.

| split | objects | views | classes |
|---|---|---|---|
| FIT | 40 | 1,640 | 8 |
| DEV | 24 | 984 | 8 |
| FRESH | 16 | 656 | 8 |

FIT is the only split that may build, fit or choose anything. DEV is for development reads
and for the failure attribution. FRESH is not touched until the Mission 21 registration is
committed.

## 3. The tasks

All four are *image* tasks: a query image is compared with every gallery image and the
gallery is ranked. Correspondence-level numbers (node top-1, geo AUC) are reported for the
tasks where a correspondence exists — T0, T1 and T3 — and **never** for T2, where the
positive is a different instance and no place-to-place truth exists.

| task | query | gallery | truth | what it tests |
|---|---|---|---|---|
| **T0** | `R(view v of object o, 15°)` — the image itself, rotated by the known transform | the same split's objects at view `-000-000`, mixed classes | object `o` | **the aggregation control.** Content is identical; only the frame moved. Any loss here is score arithmetic, not recognition. |
| **T1** | view `-022-000` of `o` | objects of `o`'s class **including** `o`, at view `-000-000` | object `o` | other view, same object, thin gallery (5 FIT / 3 DEV / 2 FRESH images before enlargement) — the placement machinery's home ground. |
| **T2** | view `-022-000` of `o` | all objects of the split at `-000-000`, mixed classes; instances of `o` excluded as positives | **the class** of `o` | other instance, same class: appearance transfer between differently-shaped objects of one class (apple/pear/tomato, cow/dog/horse). |
| **T3** | view `-022-000` of `o` | every object of the split at `-000-000`, mixed classes | object `o` | other view **in a mixed gallery**: the same query as T1 with the distractors that make 8-way retrieval a recognition problem rather than a pairwise one. |

View keys are the corpus's own: `-022-000` is a 22.5° azimuth step at elevation 000 from
the reference view `-000-000`.

## 4. Metrics, fixed now

* **node top-1** — the fraction of query places whose best partner is the true
  corresponding place; reported with its n and its per-image spread.
* **image rank-1 / MRR** — the identity score's ranking of the gallery, reported with its n
  and its per-image spread.
* **shortlist recall@S** — every shortlist number is reported **with** its recall. A
  shortlist size with no recall is not a result.
* Regimes are never pooled: T0–T3 are separate rows, and FIT / DEV / FRESH are separate
  columns.

Gallery size and the confusable-class subset are chosen **on FIT** (item 6) under the
registered constraint that the shipped score's FIT top-1 lands in [0.30, 0.85]. That choice
is a parameter of the read, recorded with its provenance; the task definitions above are
not.
