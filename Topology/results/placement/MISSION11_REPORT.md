# Mission 11 — put placement into the deployed pair score, and test recognition against the shipped matcher

Branch `arena/c316ca3a-percept6`, from `8225efc`. This report is written as the runs
land; the last section says what is still running and what it costs.

**The short version.** The deployed pair score (`JointMCS.score`) is computed after
alignment from raw node similarity and knows nothing about the placement evidence
that produced the correspondence — which is why every Mission-10 retrieval number had
to be a bespoke score. Step 1 puts one learned term inside the matcher, dark:
`score' = score_shipped + w · E`, where `E` is the mean placement credit over the
mapped pairs the frozen estimator could actually judge, shrunk by the number judged
so it cannot become a coverage statistic. `w` is fitted on FIT only. The
maximum-likelihood fit turns out to be **degenerate** — it climbs to its bound,
because the shipped score already ranks the FIT truth first on 11 of 14 queries, so
scaling the logits up sharpens them rather than reordering them. That is a
temperature being fitted, not a weight; the report says so and applies a declared
tie-break (FIT top-1, ties to the smaller weight) which lands on **w = 0.375**.

---

## 1. Step 1 — the term, the definition, and the weight

### The formula, written before the fit

For a pair (A, B) with the final mapping M:

```
judged pair   (u,v) in M with at least one slot slots_for_candidate(...) can check
pair credit   c(u,v) = plus_score(u,v) - 0.5          (the estimator's own [0,1] map)
evidence      E = sum over judged pairs of c  /  (n_judged + k),   k = 8
pair score    S' = S_shipped + w * E
```

* **centred so that no evidence is 0** — `plus_score` is centred at 0.5 by
  construction, so a pair with nothing to say contributes 0 and a negative E is
  real evidence against.
* **restricted to judged pairs** — a pair with no checkable slot contributes nothing
  and is not counted; `n_judged` and `share_judged = n_judged / |M|` are exposed on
  the result.
* **coverage-neutral** — the denominator is the number judged, not the number of
  mapped nodes. Mission 10 measured that a sum over correspondences is a coverage
  statistic (mapped pairs ran 606–921 across candidates with flat per-pair quality);
  that is exactly the failure this denominator exists to prevent.
* **shrunk** — `+ k` with k = 8 means one judged pair can move E by at most 1/9 of
  its own value.

Everything is dark: `PLACEMENT_SCORE_EVIDENCE` defaults off, and with it off `score`
is bit-for-bit the shipped number (`score_shipped` carries the same value).

### The weight, and the fact that maximum likelihood does not identify it

196 FIT pairs (14 FIT scenes × 14 candidates; irregular at in-plane 12°, out-of-plane
at y 20°). The ML objective is the retrieval likelihood the mission asks for:

```
w* = argmax_w  sum_q log softmax_c( S_qc + w * E_qc ) [truth]
```

| w | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 | 3 | 10 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| log-likelihood | −36.25 | −36.19 | −36.12 | −35.98 | −35.72 | −35.21 | −34.69 | −31.25 | −17.46 |
| FIT top-1 | 0.786 | 0.857 | 0.857 | **0.929** | 0.857 | 0.643 | 0.643 | 0.643 | 0.571 |

The likelihood is **monotone increasing in w** all the way to any bound. It has to
be: the shipped score already ranks the truth first on 11 of 14 FIT queries, so
multiplying the logits by anything makes those probabilities sharper and raises their
likelihood, while the queries it gets wrong are already wrong at w = 0. Maximum
likelihood is choosing a temperature, not a weight — it is unidentified at the
boundary in this model class, and saying otherwise would be a nicer story than the
data supports.

So the code applies a **declared tie-break**: when ML lands on the bound, take the
weight that maximises FIT top-1 on the same shortlists, ties to the smaller weight.
That is **w = 0.375** (top-1 0.929 against 0.786 at w = 0). Both numbers, the whole
likelihood curve and the top-1 curve are in `m11_fit_w.json` and
`matching/placement_score_params.json`, so the choice can be audited.

Scale check, because a weight is meaningless without one: on the FIT shortlists E has
SD 0.018 and the shipped score's candidate-to-candidate SD is 0.023 — the same order
of magnitude, which is why w = 0.375 is a refinement of the score rather than a
replacement for it.

### Defaults

Flag off, a recorded Mission-10 view-pair reproduces **per node** in both arms
(0.5934 / 0.5824, 91 ambiguous nodes each), and the three suites pass (17 + 12 + 17).
`PLACEMENT_SCORE_EVIDENCE`, `_SHRINK` and `_WEIGHT` are all in `_FRONTEND_SWITCHES`
and therefore in the switch digest every matcher-result cache key folds in.

---

## 2. Step 5 — what limits real-image coverage, measured link by link

Eight fresh ETH-80 apple pairs (reference view vs `-022-270`), one row per pair in
`m11_coverage.json`. The chain that produces a judged pair, link by link:

| link | measured |
|---|---|
| anchors | 100 % |
| anchor coherence ≥ 0.5 at the single 7×7 window | **40.6 %** (median coherence 0.423; **59.3 % of anchors are below the bar**) |
| nodes with ≥ 1 payload edge | **31.7 %** (10.9 incident edges per node) |
| nodes the aligner mapped | 79.6 % |
| **mapped pairs with ≥ 1 judged slot** | **42.6 %** |
| payload edges with both endpoints mapped | 88.9 % |
| place spacing on real images | 5.77 px (synthetic: 6.32 px) |

The limiting link is the **frame provider**, not the tolerance and not the radius:
59 % of real anchors never clear the coherence bar at the 7×7 window, so they never
get a payload edge, so their pairs can never be judged. The module already ships the
multi-scale provider for exactly this reason (`field_multiscale` /
`node_frames_multiscale`, smallest window that clears the bar).

### The fix, shipped dark

`PLACEMENT_FRAME_MULTISCALE` (dark, in `_FRONTEND_SWITCHES`) switches `build_edges`
onto that provider. Same eight pairs, flag off → on:

| quantity | off | on | Δ |
|---|---|---|---|
| nodes with a frame | 40.6 % | **50.4 %** | +9.8 pts (+24 % relative) |
| nodes with ≥ 1 payload edge | 31.7 % | **34.9 %** | **+3.1 pts** |
| incident edges per node | 10.9 | 13.4 | +2.5 |
| **mapped pairs judged** | 42.6 % | **44.9 %** | **+2.3 pts** |
| match time per pair | 24.0 s | 25.8 s | +7.6 % |

**The frame fix does not translate one-for-one, and that is the finding**: +9.8 points
of frames buy only +3.1 points of nodes with an edge, because the next limit is
target availability (`TARGET_K = 16` within `4 × scale`, and a node whose neighbours
all lack frames has nothing to point at), and the judged-slot definition after that.
The flag is dark and **uncredited** — it changes the payload geometry, so believing it
needs a fresh read, which this mission did not spend.

### The pear inversion

(to be filled by `m11_pear.json`)

---

## 3. Step 3 — the synthetic retrieval read

(to be filled)
