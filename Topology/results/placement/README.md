# The placement system, for someone who has not read the other thirty reports

This folder is the evidence shelf for the object-placement work (Missions 1-22).  This file is the
entry point: what the shipped thing is, how to run it, what its numbers mean, and what it does not
do.  Everything here is measured on **ETH-80** (8 classes x 10 instances, one view per instance per
pose bin) and every rate below carries its **n** and its **chance**.

## 1. What the system does

Given a query photograph of one object and a gallery of one photograph per known object, return a
ranking of the gallery.  Three layers, each cheaper than the next is accurate:

```
image -> graph (perception pipeline) -> PLACES  (nameable parts with a configuration)
                                          |
     FRONT DOOR  (cheap, no matching)  ->  shortlist of S = 8 objects
       appearance      4x4x4 colour + 12-bin gradient histogram, cosine
       place voting    every query place probes a pooled content index and votes for the
                       gallery object its partner came from
       combined        the two lists rank-fused (reciprocal rank fusion, k = 60, no weight)
                                          |
     MATCHER  (expensive, per pair)  ->  re-orders the 8 by the shipped pair score
                                          |
                                      ranking of the gallery
```

The matcher is the Mission-19 system and is unchanged.  The front door is Mission 22's addition:
it makes the matcher look at 8 objects instead of all of them, at about 1.5 % of the matcher's
cost.  **The rerank never removes a candidate**, so the assembled system's ceiling is the front
door's recall@S, and both numbers are always reported together.

## 2. How to run it

Everything reads the deployed configuration, which is switched on by an environment flag:

```bash
export PLACEMENT_M19=1                      # the deployed matcher

# 1. the front doors on a split (writes a JSON with recall@S, top-1, chance, per class, per image)
.venv/bin/python -u Topology/placement_m22_eth80.py --split FIT --stage frontdoor --workers 2

# 2. the assembled system: shortlist -> matcher -> reorder (resumable; it keeps a pairs cache)
.venv/bin/python -u Topology/placement_m22_eth80.py --split FIT --stage assemble --workers 2 \
    --front fused --tag m22_assemble_fit_fused

# 3. the audit: does every place carry position, scale and orientation?  (read-only)
.venv/bin/python -u Topology/placement_m22_audit.py --split FIT --images 3 --pool 6

# 4. the power arithmetic behind the registration (no images, no graphs)
.venv/bin/python -u Topology/placement_m22_power.py
```

Splits are fixed in `m20_splits.json` (salt `M20-split-v1-2026-11-18`, rule
`sha256(salt|class|instance)`): **FIT 40, DEV 24, FRESH 16 objects**.  FIT is where choices are
made, DEV reports, FRESH is held out.  Mission 22 read FIT and DEV and did **not** read FRESH --
see `MISSION22_PREREGISTRATION.md` §4 for the power arithmetic that stopped it.

The three switches the front door and the rerank add are **off by default**, in
`parameters/_text.py`:

| flag | default | what it turns on |
|---|---|---|
| `PLACEMENT_FRONTDOOR` | `off` | which door a harness may build a shortlist with (`appearance`, `voting`, `fused`) |
| `PLACEMENT_ASSEMBLED` | off | the matcher re-orders a front door's shortlist |
| `PLACEMENT_CLASS_SCORE` | off | a class is scored by pooling its instances' matcher scores |
| `PLACEMENT_VOTE_HITS` | 3 | pool places one query place may vote with |
| `PLACEMENT_VOTE_ROT` | on | whether a place with an orientation frame also votes on rotation |

With all of them off, nothing in the tree calls the new code and the deployed numbers are
byte-identical to the Mission-20 record (checked: `placement_m12_perf.py --check`, 0 diffs).

## 3. How to read a number here

* **recall@8** counts a hit when the truth is inside the 8 returned.  Its chance is **8/N**, not
  1/N: a random shortlist of 8 of N contains the truth with probability 8/N.  (Mission 21's tables
  printed the top-1 chance beside recall@8; that column was wrong and Mission 22 corrects it.)
* **top-1** chance is **1/N**.  On the FIT gallery (40) that is 0.025; on DEV (24), 0.042.
* **class top-1** chance is **k/N**, k = the gallery items of the query's class (5 of 40 on FIT,
  3 of 24 on DEV) -- a random top-1 lands on the right class with probability k/N.
* **n** is objects, never pairs.  A number without an n is not a measurement.
* Timings are seconds per query, on this box (2 cores, 3.9 GB), with the graph cache warm.

## 4. What it does not do

* It does not recover the object's pose.  The place frames carry an axial orientation, but the
  correspondences the matcher chooses between two views **do not agree on a rotation** (measured:
  interquartile range 2.8 rad, i.e. the whole [0, pi) circle; `m22_placeproof.json`), so the
  transform histogram votes for nothing and the door counts instead of clustering.  That is why
  the shipped door is the *counted* vote, not the geometric one.
* It does not work at a distance.  The operating range, as far as it has been measured, is
  in-plane rotation to 15 degrees and pitch to 20 degrees (descriptor-level diagnosis); **yaw is
  unresolved**: the accuracy harness shows no yaw-specific loss at 20 degrees (0.234 vs 0.203
  in-plane), which contradicts the earlier probe, and the pitch cell that would settle it has not
  been run through that harness.
* It does not answer "is this the same object" better than chance on the 40-item gallery at
  top-1 with the matcher alone: shipped matcher top-1 is 0.175 (7/40, chance 0.025) and the
  assembled system reaches 0.425 (17/40) because the front door puts the answer inside the 8.
* The FRESH split is unread: there is **no confirmatory number** in this folder.  FIT and DEV
  are the splits the mechanism was chosen on.

## 5. Where the evidence is

| file | what it holds |
|---|---|
| `MISSION22_REPORT.md` | the current mission: the three builds, the gate, the numbers, the verdict |
| `MISSION22_PREREGISTRATION.md` | the arms, the primaries, the power arithmetic, and why FRESH was not read |
| `MISSION21_REPORT.md` | the frozen ruler, the coverage control, the yaw question |
| `MISSION20_REPORT.md` | the splits, the tasks, the shipped matcher's record |
| `m22_fit.json`, `m22_dev.json` | front doors: recall@8, top-1, chance, per class, per image |
| `m22_assemble_fit_*_read.json` | the assembled system and the class score, per arm |
| `m22_audit.json`, `m22_placeproof.json`, `m22_power.json` | the audit, the diagnosis, the arithmetic |

Every result JSON is written through `matching/m15_manifest.write_result`, so it carries the
manifest of the flags that were in force when it was produced.
