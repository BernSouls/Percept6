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
