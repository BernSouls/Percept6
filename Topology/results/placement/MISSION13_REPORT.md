# MISSION 13 — fix the rotation failure, then read where the question can be asked

Branch `arena/c316ca3a-percept6`, from tip `71212f5`.

## STEP 1 — the frame provider: fallback, sub-pixel sampling, adoption

### 1a. The fallback was the worst of the three providers

`node_frames_keypoint_then_field` fell back to the **multiscale ladder** wherever the
keypoint estimator had no frame. The ladder is the worst of the three sources under
rotation, and the node-level measurement now shows it loading its deficiency onto the
composition:

| provider (node level, 12 images × delta ∈ {5°, 15°} = 24 cells, 14 110 anchors) | share ≤3° | MAD | frames kept |
|---|---|---|---|
| shipped single 7×7 field, nearest pixel | 0.673 | 1.54° | 0.388 |
| **single 7×7 field, sub-pixel (1c)** | **0.718** | 1.29° | 0.391 |
| multiscale ladder (7/15/23/31) | 0.616 | 1.73° | 0.509 |
| keypoint, **smallest** clearing σ (deployed) | 0.563 | 2.35° | 0.609 |
| keypoint (smallest σ) + field fallback | 0.558 | 2.42° | 0.621 |
| keypoint (smallest σ) + multiscale fallback (M12) | 0.550 | 2.65° | 0.645 |
| **keypoint at the LARGEST σ (σ = 4), bilinear** | **0.811** | **0.70°** | 0.359 |
| keypoint at σ = 4 + field fallback | 0.735 | 1.29° | 0.465 |

Two things fall out, and the second one is the reason M12's number was wrong:

* **a fallback always dilutes.** keypoint+field (0.558) < keypoint (0.563) <
  keypoint σ=4 + field (0.735) < keypoint σ=4 (0.811). A provider that adds *worse*
  frames to *better* ones lowers the pooled equivariance, and here it does not even
  buy judged pairs (σ = 4 with the fallback: 74.5 judged and 0.1221 share, against
  75.1 and 0.1231 without it). The fallback is now `"none"` by default.
* **M12 measured a provider the tree does not run.** M12's "keypoint provider",
  reported at 0.803 synthetic / 0.913 real, sampled the tensor at
  `max(KEYPOINT_SIGMAS) = 4.0`. The deployed `node_frames_keypoint` takes the
  **smallest** σ that clears the coherence bar, which is **0.563** — *worse than the
  field it was supposed to replace*. The σ rule is now an explicit, cache-keyed
  parameter (`PLACEMENT_FRAME_KEYPOINT_SIGMA`), and the adopted provider is the one
  that was actually measured: **largest σ, no fallback**.

### 1b. The adoption rule, stated before the measurement

Declared in `Topology/placement_m13_frames.py` in commit `0a1a28b`, before any number
existed (the second candidate set in `5dc0ffd`, also before it was run):

* **E (equivariance)** — pooled share within 3° over the 24 cells, paired
  scene-clustered bootstrap (4000 draws, 12 clusters) against the shipped 7×7 field;
* **C (coverage)** — the deployed matcher on the 8 hard FIT pairs: mean
  `placement_share_judged` and mean judged count;
* **eligible** ⟺ C is no worse than the shipped field on *both* C numbers;
* **adopted** ⟺ highest E among the eligible, and only if the paired E gain over the
  shipped field excludes 0 at 95 %.

M12's withdrawn test (an 8-pair near-tie *accuracy* interval of [+0.000, +0.188]) is
not used anywhere in this decision.

| candidate | E | E gain vs shipped field [95 %] | C: judged share / count | eligible |
|---|---|---|---|---|
| **keypoint σ = 4, no fallback** | **0.811** | **+0.138 [+0.100, +0.183]** | **0.1231 / 75.1** | **yes** |
| keypoint σ = 4 + field fallback | 0.735 | +0.062 [+0.037, +0.087] | 0.1221 / 74.5 | yes |
| field, sub-pixel | 0.718 | +0.045 [+0.022, +0.074] | 0.1029 / 62.8 | yes |
| field, nearest (shipped) | 0.673 | 0 (reference) | 0.1029 / 62.8 | — |
| multiscale | 0.616 | −0.057 [−0.080, −0.034] | 0.1222 / 74.5 | yes |
| keypoint, smallest σ | 0.563 | −0.110 [−0.156, −0.058] | 0.1233 / 75.2 | yes |
| keypoint (smallest) + field | 0.558 | −0.115 [−0.159, −0.067] | 0.1236 / 75.4 | yes |
| keypoint (smallest) + multiscale | 0.550 | −0.123 [−0.167, −0.075] | 0.1248 / 76.1 | yes |

> **ADOPTED: keypoint at the largest σ (4.0), no fallback** — E 0.811 against the
> shipped 0.673 with a paired interval that excludes zero, coverage no worse
> (judged share 0.1231 vs 0.1029, judged count 75.1 vs 62.8), and a frame MAD of
> 0.70° against 1.54°. It is reached through the dark switch
> `PLACEMENT_FRAME_KEYPOINT` (default off), with
> `PLACEMENT_FRAME_KEYPOINT_SIGMA = "largest"` and `PLACEMENT_FRAME_FALLBACK = "none"`.

The cost is stated rather than buried: σ = 4 keeps a frame on **0.359** of anchors
against the field's 0.388 — a smooth tensor is less often coherent — so the win is
precision, and the judged share is what shows it does not cost coverage in practice.

### 1c. Sub-pixel frame sampling

`frame_at` rounded a sub-pixel anchor to the nearest pixel. It now samples
bilinearly, with the axial angle interpolated on its **doubled angle**
(`cos 2θ, sin 2θ` → `½ atan2`) so the π-periodic wrap cannot tear a 179°/1° pair into
90°. Dark by default (`PLACEMENT_FRAME_SUBPIXEL`), cache-keyed.

Measured on the same 24 cells: equivariance **0.673 → 0.718** (+0.045
[+0.022, +0.074]), MAD 1.54° → 1.29°, frames kept 0.388 → 0.391. On the σ = 4
provider it is inert by construction (that path is already bilinear).

### Defaults

All three new switches are off or inert at their defaults, and the tree's own fixed
harness agrees **exactly** with the pre-step baseline: 14 pairs × 2 arms, **0 numeric
diffs** (score, shipped score, evidence, judged count — 112/112 identical) and **0
mapping diffs**. The graph cache keys moved (three new names in `_FRONTEND_SWITCHES`),
as the tree's rule requires for a switch that can change a payload.

Artifact: `Topology/results/placement/m13_frames.json` (2 328 equivariance rows,
96 coverage rows, both decisions).

## STEP 2 — decomposing the rotation collapse: detection, not appearance

6 irregular synthetic scenes + 6 real photographs, poses 5°, 15°, 1.15×, plus the
**exact-shift control** (an integer shift resamples nothing, so it validates the
instrument). Every number below is on all nodes/anchors inside the image, not on a
bank.

| pose | detection ≤1 px | detection ≤2 px | truth survives | appearance-only acc (raw) | appearance-only acc (frame-canonicalised) |
|---|---|---|---|---|---|
| **shift +8 px (control)** | **0.469** | 0.507 | 0.111 | 0.782 | 0.751 |
| rot 5° | **0.129** | 0.260 | 0.027 | 0.331 | 0.320 |
| rot 15° | **0.170** | 0.296 | 0.045 | 0.295 | 0.392 |
| scale 1.15× | **0.103** | 0.240 | 0.020 | 0.176 | 0.149 |

* **a. APPEARANCE is not the cause.** For the truth pairs that do survive, the
  signature similarity of the true pair (1.0 at zero pose by construction) reads
  0.33 / 0.30 / 0.18, and an *appearance-only* correspondence (is the true
  counterpart the argmax over the whole same-class reference set?) reaches
  0.331 / 0.295 / 0.176 — far above the correspondence the whole pipeline achieves.
  The orientation-slot hypothesis was tested directly: every circular slot whose
  name carries "theta" (`CIRCULAR_CYCLE_LENGTHS`: ORIENTATION slot 0,
  CONTOUR_PEAK slot 0/4, CONTOUR slot 1, CORNER slot 1, …) was rotated into the
  node's own frame, with the matching population axis rolled by the same bins.
  Canonicalising them moves appearance-only accuracy by **−0.011 (5°), +0.097
  (15°), −0.027 (1.15×)** — mixed, small, and nowhere near enough to explain a
  collapse from 0.78 to 0.18. **The orientation slots are not the dominant term.**
* **b. DETECTION is.** Only 13 % (5°), 17 % (15°), 10 % (1.15×) of nodes are
  re-detected within 1 px of their transformed anchor, against 47 % under the exact
  shift — and the shift control's own median nearest-node distance is **0.00 px**
  with a median residual vector of (0.00, 0.00), so the instrument's transform is
  exact and the loss is real.
* **c. TRUTH is a consequence of (b), not an independent term.** Only 2.7 / 4.5 /
  2.0 % of nodes survive the tight truth rule (1 px nearest, runner-up 2× further,
  mutual) because most nodes have no neighbour to pair with at all.

**The mechanism inside (b) is positional quantisation, not detector noise.** The
shipped front end writes node anchors on an image-aligned lattice: 39 % of synthetic
anchors sit exactly on multiples of 8 px and 84 % on integers (real photographs:
54 % / 84 %). A rotation or a scale moves every physical point *off* that lattice, so
the nearest re-detected node is a lattice step away — median 3.6 px, and the
detection curve (0.05 at 0.5 px → 0.13 at 1 px → 0.26 at 2 px → 0.38 at 3 px) has
exactly the shape a constant lattice offset produces. It is not sub-pixel jitter:
the residual vector is unbiased ((+0.10, −0.05) px) and the distances are ~half a
lattice step.

## STEP 3 — the fix: sub-pixel anchors, and what it buys

The lever the tree already has is sub-pixel anchoring (`FE_SUBPIXEL_PEAK_POS`,
`FE_SUBPIXEL_GABOR_ANCHOR`, both dark). Turning them on takes the anchored share of
nodes off the lattice (on-lattice 0.39 → 0.01 synthetic, 0.54 → 0.005 real;
non-integer share 0.14 → 0.45, 0.16 → 0.64), which is precisely the dominant term.

Measured on the same 12 images × 3 poses, **arm A = the shipped aligner with
placement OFF**, scored on a criterion that does not depend on the flags: for every
reference node inside the image, is the mapped partner's anchor within T px of the
transformed anchor? (scene-clustered bootstrap over the 12 images, 4000 draws)

| pose | detection ≤1 px off → on | arm A ≤1.5 px off → on | gain [95 % CI] |
|---|---|---|---|
| 5° | 0.129 → 0.177 | 0.047 → **0.064** | **+0.017 [+0.008, +0.027]** |
| 15° | 0.170 → 0.343 | 0.045 → **0.093** | **+0.048 [+0.040, +0.060]** |
| 1.15× | 0.103 → 0.177 | 0.039 → **0.073** | **+0.034 [+0.023, +0.044]** |
| real photos only, 5°/15°/1.15× | 0.090→0.128 / 0.149→0.365 / 0.089→0.160 | 0.042→0.058 / 0.036→0.087 / 0.030→0.056 | — |

> **The floor is left, and the size of the move is stated, not celebrated.** Arm A's
> geometric correspondence under 5°, 15° and 1.15× roughly doubles (+36 %, +107 %,
> +87 %), every interval excludes zero, and the exact-shift control is untouched at
> 0.78 appearance-only / 0.47 detection. It is still 0.06–0.09, not 0.76: the
> rotation regime remains a different regime, and Step 5 registers it as such
> rather than pooling it with the working one.

**What remains, named and unclaimed.** Arm A reaches only **27–37 % of its own
detection ceiling** (0.064/0.177 at 5°, 0.093/0.343 at 15°) — i.e. when a counterpart
does exist, the aligner finds it less than half the time. Two further candidates were
tested and eliminated in the same pass: the vector-address seed (`addrs_a/addrs_b`
emptied) changes arm A's accuracy by **0.000 on all 18 real-photo cells**, so the
seed is not it; and the appearance term is not it either (Step 2a). That residue is
the next piece of work, and it is a matcher change, not a flag.

**Adopted (dark):** the M13 read arms run with `FE_SUBPIXEL_PEAK_POS = True` and
`FE_SUBPIXEL_GABOR_ANCHOR = True`; both remain **off by default**, so the shipped
tree is unchanged (Step 1's harness: 112/112 numbers and 28/28 mappings identical to
the pre-step baseline).

Artifacts: `m13_decompose.json` (48 cells), `m13_repeat.json` (72 cells),
`m13_repeat_boot.json`.

## STEP 4 — refitting w by pairwise margin: unidentifiable, keep the shipped weight

The softmax's `T` is a pure sharpening knob and its optimum lives at a corner; that is
already established and is not re-derived here. This step removes `T` (fixed at the
deployed 1.0) and fits **w alone** to a Bradley–Terry **pairwise margin** likelihood
over Mission 12's own bank (`m12_fit_w_progress.jsonl`, 252 queries × 4 candidates =
756 ordered (truth, rival) pairs):

    m(t, c) = ( B(t) - B(c) ) + w · ( E(t) - E(c) )
    ll(w)   = Σ_pairs log σ( m(t, c) )

`B` is the deployed shipped score, `E` the placement evidence on the STEP-2 scale.
A pairwise margin sees only differences, so it cannot be inflated by the score's level
— which is exactly what the softmax was exploiting.

**Strata cut before fitting** (the FIT view grid, no view chosen by looking at w):
S1 in-plane easy (5/10/15°), S2 in-plane hard (25/30/35°), S3 out-of-plane
(y10/20/30, x10/20), S4 scale (s09/s115). Each fitted alone, plus the pool.

| fit | pairs | ŵ | interior maximum | profile-w interval (Δ2ll ≤ 3.841) | P(ΔE > 0) | corr(ΔB, ΔE) |
|---|---|---|---|---|---|---|
| pooled, FIT (¾) | 567 | **3.00 (bound)** | **no** | [2.75, 3.00] | 0.735 | 0.811 |
| pooled, all | 756 | 3.00 (bound) | no | [2.81, 3.00] | 0.745 | 0.785 |
| S1 in-plane easy | 126 | 3.00 (bound) | no | [2.37, 3.00] | — | 0.863 |
| S2 in-plane hard | 126 | 3.00 (bound) | no | [1.44, 3.00] | — | 0.786 |
| S3 out-of-plane | 210 | 3.00 (bound) | no | [2.34, 3.00] | — | 0.742 |
| S4 scale | 42 | 3.00 (bound) | no | [0.97, 3.00] | — | 0.409 |

**There is no interior maximum, and the reason is measurable, not mysterious:** `E`
and `B` agree in sign on 73–84 % of pairs (corr 0.81), so raising w raises almost every
margin monotonically — the bank is (nearly) separable in the `E` direction and the
likelihood has no finite maximiser. Every stratum repeats the pattern.

**Held-out margin, and why it does not license adoption:**

| w | held-out mean margin | held-out top-1 |
|---|---|---|
| 0 | 0.0332 | **0.7937** |
| **0.375 (shipped)** | **0.0419** | **0.7937** |
| 1.0 | 0.0557 | 0.7619 |
| 2.0 | 0.0758 | 0.7143 |
| 3.0 (ŵ) | **0.0949** | 0.6984 |

The margin improves by +0.062 from w = 0 to ŵ — and the ranking **falls** by 9.5
points. A margin that grows while the order degrades is scale inflation, not a better
weight: the objective is unbounded in exactly the direction that hurts.

**The decisive subset.** Only the 119/756 pairs where `ΔE` opposes `ΔB` can identify w.
On them the profile is flat across the whole grid (ll = −81.213 at w = 0 vs −81.319 at
w = 0.375, interval [0, 3.00]) and the accuracy is *maximal at w = 0.25* (0.815) and
falls monotonically to 0.639 at w = 3 — i.e. the informative pairs reject the direction
the pooled profile runs in.

> **Verdict: w is not identifiable by a pairwise margin on this bank. The shipped
> weight w = 0.375 is KEPT** — it sits at the top of the flat held-out region
> (0 ≤ w ≤ 0.375, all at 0.7937) and within a hair of the informative subset's own
> optimum. Adopting ŵ = 3.00 would be adopting a boundary artefact.

Artifact: `m13_margin.json`.

## STEP 5 — the real-photo read, re-registered by regime, taken once

Registration signed at `m13_prereg_read.json` **before the read** (regimes, arms,
contrast, per-stratum MDE, verdict rule, the BROKEN declaration for the secondary
strata, and the one-look rule). 6 validation photographs (the package-bundled set —
ETH-80 is still absent, Step 6), 4 transforms = 24 cells, all graphs built with the
Step-3 sub-pixel anchors ON. Arms A = placement OFF, B = placement ON/evidence OFF,
C = placement ON/evidence ON at the shipped w = 0.375. Contrast **C − A**, photo-
clustered bootstrap (4000 draws). **Nothing is pooled**: every stratum is read and
verdicted alone.

| stratum | role | near-tie bank | acc A | acc B | acc C | C − A [95 % CI] | MDE | verdict |
|---|---|---|---|---|---|---|---|---|
| **trans8 (exact shift)** | **PRIMARY** | 98 | 0.653 | 0.663 | 0.653 | **0.000 [0.000, 0.000]** | 0.200 | **NULL_SUPPORTED** |
| rot05 | SECONDARY | 9 | 0.556 | 0.556 | 0.556 | 0.000 [0.000, 0.000] | 0.660 | BROKEN (as declared) |
| rot15 | SECONDARY | 28 | 0.143 | 0.071 | 0.071 | −0.044 [−0.100, 0.000] | 0.374 | BROKEN (as declared) |
| scale115 | SECONDARY | 8 | 0.125 | 0.125 | 0.125 | 0.000 [0.000, 0.000] | 0.700 | BROKEN (as declared) |

Unambiguous bank (reported alongside, not instead of the registered one):
trans8 A 0.741 / B 0.724 / C 0.724 (n = 297); rot05 0.272 (n = 81); rot15 0.255
(n = 161); scale115 0.211 (n = 71).

**Plain verdict.** In the regime where correspondence works (exact shift), the
placement evidence + weight channel changes **not one mapping** — C − A is exactly
0.000 in all six photographs, and on the unambiguous bank placement ON is mildly
**harmful** (0.741 → 0.724, from 2 of 6 photos: moon 46→44, gravel 66→64). In the
rotation and scale regimes the read confirms the Step-3 state instead of repairing it:
the strata are declared BROKEN and stay BROKEN, and the intervals there carry no
information either way (banks of 8–28 nodes; the near-tie bank is small *because* the
sub-pixel anchors leave few appearance-ambiguous nodes, which is a coverage number,
not evidence about the mechanism).
**No claim is made about rotation or scale from this read, and none is made from a
pooled interval.**

Honest power note: the registered near-tie banks are thin (8–98 nodes → MDE 0.20 on
the PRIMARY, 0.37–0.70 on the SECONDARY), so the PRIMARY null is a null at MDE 0.20,
not a null at 0.05; the secondary verdicts rest on the *declared* BROKEN state, not on
these intervals. That is stated rather than papered over.

Artifacts: `m13_prereg_read.json`, `m13_read_real.json`, `m13_read_real_progress.jsonl`.

## STEP 6 — ETH-80: still blocked, no numbers invented

`data/ETH-80-master` is a symlink to `.cache/ETH-80-raw` and contains **25 directories
and 0 files** (11 + 11 + 3 + 0 subdirectories, `4/` absent): the tree was created but
the tarball never finished (it stalls at 322 534 399 B) and pooch is unreachable from
this sandbox. There is therefore **no ETH-80 result in this mission**, and none is
invented. The real-image set stays the package-bundled photographs with their
licence manifest (`m12_real_photos.json`), which is what Steps 2 and 5 used.

## CLOSING — state of the tree

* **Defaults byte-identical.** `placement_m12_perf.py --save` then `--check` in a fresh
  process: **0/112 numeric diffs, 0/28 mapping diffs, `identical: True`**; wall
  4.61 s (arm A) / 5.22 s (arm C) per pair against 4.44 / 5.07 in the Step-1 baseline.
  Everything this mission added to the shipped path is behind a switch that is **off**
  (`PLACEMENT_FRAME_FALLBACK="none"` and `PLACEMENT_FRAME_KEYPOINT_SIGMA="largest"` are
  the adopted defaults; `FE_SUBPIXEL_PEAK_POS` / `FE_SUBPIXEL_GABOR_ANCHOR` stay off
  and are turned on only by the M13 read arms).
* **The `attachment: mapping` "nondeterminism" was a measurement bug, now fixed.** The
  checker compared the JSON-round-tripped `mapping` (a list of `[int, int]` lists) with
  the fresh run's list of `(int, int)` tuples, so *every* cell reported a diff (28/28)
  while every number matched. `_norm()` in `placement_m12_perf.py` normalises before
  comparing, and the run is then bit-identical. Mappings were never nondeterministic.
* **Suites pass:** `Topology/test_placement.py` 17 tests OK, `Topology/test_placement_m3.py`
  12 tests OK, `python -m unittest discover -s tests` 17 tests OK.
* **What moved, in one line:** the frame provider is fixed (+0.138 equivariance at
  equal judged coverage), the collapse is attributed to **detection / anchor
  quantisation** rather than appearance, that term is repaired to the extent the tree
  allows (sub-pixel anchors: arm A under 5°/15°/1.15× goes 0.047/0.045/0.039 →
  0.064/0.093/0.073, all intervals excluding zero), w is shown to be **unidentifiable**
  and left at the shipped 0.375, and the real-photo read is re-registered by regime and
  taken once: the working regime says **no effect at all** (C − A = 0.000), the broken
  regimes stay broken by declaration and by measurement.
