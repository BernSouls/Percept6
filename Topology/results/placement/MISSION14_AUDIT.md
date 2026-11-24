# MISSION 14 — AUDIT: the system against the goal

**Scope.** No confirmation reads were run. Everything below is code read at
`arena/c316ca3a-percept6` (tip `6cf69fc` + this mission's one fix) or a *labelled probe*
on **already-cached graphs** (probes answer only the two questions the brief allows:
"can w·E flip a decision" and "where is a detected pair lost"). Probe scripts were
throwaway (`/tmp/probe_*.py`); every number they produced is quoted with the condition
that produced it, and none of them is a mission result.

Severity words are the brief's: **BLOCKS THE GOAL**, **DEGRADES**, **COSMETIC**.

---

## B1 — FRONT END

### B1-1 BLOCKS THE GOAL — a node is a (lens, signature) CLASS, not a place

`core/place_key.py:88-104`

```python
def salted_sig_hash(lens: str, sig_hash: bytes, anchor: Any) -> bytes:
    if not place_layer_on() or not is_place_lens(lens):
        return sig_hash
    ...

def node_key(lens: str, sig_hash: bytes, anchor: Any) -> Tuple[str, bytes]:
    """The key a mint site should use."""
    return (str(lens), salted_sig_hash(lens, sig_hash, anchor))
```

`parameters/_text.py:1207` — the salt that would make the key place-based is **dark**:
`FE_PLACE_LAYER: bool = os.environ.get("FE_PLACE_LAYER", "0") != "0"`.

**Goal requires** (§1): one physical feature = one node = one distinct position; two
features at the same point are not two places. **The code does** the opposite in both
directions: a node is a *signature class*, so one node stands for many places (position:
first write wins, `concept_graph.py:844-864`), and every lens that fires at one point
mints its own node.

**Measured (probe, cached graphs, default flags):**

| graph | nodes | distinct positions | positions carrying >1 node | worst |
|---|---|---|---|---|
| `syn|m11ir_00` | 668 | 373 | **165** | **12 nodes at one point** |
| `real|cell` | 421 | 234 | 87 | 9 |
| `real|grass` | 1241 | 552 | 236 | 9 |

So **44–56 % of the distinct places are carried by more than one node**, and the tree has
known this since Mission 5 (`matching/placement_edges.py:543-546`: "Census
(m5_stage0_census.json): 179 placement nodes sit at 55 places, max 16 at one site").

**Minimal fix.** `FE_PLACE_LAYER=1` is built and dark; it fixes "one node per
detection" but not "one node per place". The minimal correct fix is to **merge
co-located detections into one node carrying a LENS SET** and to keep the position as
the node's only place. Cheapest decisive test: recount places vs nodes with the merge
applied on the cached graphs (one afternoon, no read).

### B1-2 BLOCKS THE GOAL — the identity is view-fragile: the same place is a different lens after a 5° rotation

`matching/joint_mcs.py:304-318` (the only place identity is compared)

```python
def node_sim(u,v):
    x,y=sigs_a.get(u),sigs_b.get(v)
    if x is None or y is None or x[0]!=y[0]: return 0.0   # THE CLASS GATE
    return _sig_similarity(x[1],y[1])
```

**Measured (probe, 6 image × 2 pose cells, 68 detected truth pairs):**

| category | pairs | share |
|---|---|---|
| positive signature similarity | 20 | 29.4 % |
| **blocked purely by lens class** | **48** | **70.6 %** |
| blocked by same-class zero similarity | **0** | 0.0 % |

The lens classes disagree systematically: `EndStopped vs CONTOUR_PEAK` (6),
`BorderOwnership vs CONTOUR_PEAK` (4), `Boundary vs CONTOUR_PEAK` (4),
`BorderOwnership vs EndStopped` (4), `BorderOwnership vs Position` (3), …

**This is the residue Mission 13 measured as "arm A finds only 27–37 % of the
counterparts that are detected".** It is not an assignment failure and not a
pruning failure — the population is excluded before either runs.

**And it cannot be repaired by deleting the gate** (probe, same cells): comparing the
quantised slots across lens classes gives, for the 48 class-gated truth pairs, mean slot
agreement **0.080** against **0.127** for random pairs — **AUC 0.512, i.e. chance**. The
two classes' signatures are not the same measurement in different bins; they are
different measurements. Turning on the keypoint layer does not cure it either
(`FE_KEYPOINTS=1`, 3 synthetic + 3 real cells: 167 truth pairs, 11.4 % positive,
**74.3 % class-gated**, arm A correct 4/167).

**Minimal fix.** Give the place a descriptor that is *comparable across lenses and
stable across views* — which is what the placement payload already is (vectors in the
place's own frame). That is the single highest-value change in this audit; see Part D.

### B1-3 DEGRADES — anchors are quantised to the pixel and, for the tiling lenses, to an 8-px lattice

Writers, in order of authority:

```python
# orchestration/binder.py:1032-1050
def _remember_node_position(self, nid, anchor, det=None):
    pos = self._step_anchor_positions
    p = getattr(det, "_place_pos", None) if det is not None else None
    if p is None:
        if pos is None: return
        p = pos.get(anchor)
    if p is not None:
        self.cg.note_pattern_node_position(nid, p)
```

```python
# concept_graph.py:857-864  (FIRST WRITE WINS)
if nid in self._pattern_node_position:
    return
...
self._pattern_node_position[nid] = (float(position[0]), float(position[1]))
```

```python
# matching/placement_edges.py:178-184  (frame sampled at the NEAREST PIXEL)
xi, yi = int(round(float(x))), int(round(float(y)))
a, c = ang[yi, xi], coh[yi, xi]
```

Only three writers set `_place_pos`: `lenses/keypoints.py:308` (unconditional, and the
layer is dark), `lenses/base.py:2317` (under `FE_PLACE_LAYER`, dark) and
`lenses/base.py:2678` (under `FE_SUBPIXEL_GABOR_ANCHOR`, dark). At the defaults the
anchor is therefore the **region centroid**, and first-write-wins means a coarse
earlier write can never be improved by a finer later one.

**Measured (M13, given):** 39 % of synthetic anchors exactly on multiples of 8 px, 84 %
on integers; detection within 1 px collapses 0.469 (exact shift) → 0.129 (5°). This is
the mechanism M13 named. **Not a duplicate finding** — what is new here is that the
*writer order* (`first write wins`) can override a sub-pixel position that a later
detection of the same node measured, which is why turning the sub-pixel flags on only
doubles the floor instead of lifting it.

**Minimal fix.** Make `note_pattern_node_position` prefer a *higher-precision* writer
(the same rule the `_place_pos` path already uses for detections) instead of
first-wins.

### B1-4 DEGRADES — the frame provider that runs at the defaults is the one M13 measured as the worst of the three

`matching/placement_edges.py:522-530`

```python
if bool(_param("PLACEMENT_FRAME_KEYPOINT", False)):
    frames, _wins = node_frames_keypoint_then_field(
        subgraph, image_u8,
        sigma_rule=_param("PLACEMENT_FRAME_KEYPOINT_SIGMA", "smallest"))
elif bool(_param("PLACEMENT_FRAME_MULTISCALE", False)):
    frames, _wins = node_frames_multiscale(subgraph, field_multiscale(image_u8))
else:
    ang, coh = field_angles(image_u8)      # <- the default
    frames = node_frames(subgraph, ang, coh)
```

`PLACEMENT_FRAME_KEYPOINT` defaults to `False` (`parameters/_text.py:1290`), so the
shipped provider is the single 7×7 field sampled at the nearest pixel — M13 measured
0.673 equivariance against 0.811 for the adopted keypoint provider, and it loses 59 % of
real-image anchors below the coherence bar (M11). The adopted configuration
(`KEYPOINT=1`, `SIGMA=largest`, `FALLBACK=none`) is **dark and has never been used in a
read** (Part C).

Also: the frame is an **axial** angle (`0.5*arctan2(...)`, `matching/placement_edges.py:215`),
i.e. a direction modulo π with no sign. §2 asks for a **signed** direction relative to a
continuous orientation; an axial frame cannot express the difference between "v is to my
left" and "v is to my right" except through the vector's own sign, which the payload does
keep — but any consumer that compares *axes* alone (e.g. `frame_coherence`) has lost it.

**Minimal fix.** None needed for the audit: the provider exists and is measured. What is
needed is a **read** with the adopted provider (Part C, "never read together").

### B1-5 DEGRADES — absolute orientation is in the identity

`lenses/keypoints.py:263-286`

```python
def signature_of(p: Keypoint) -> Any:
    th = float(p.theta) % np.pi
    return TypedSignature(
        kind="CANONICAL",
        class_=LENS_OF_KIND.get(p.kind, "Keypoint"),
        quantized=(float(_q(th, 0.0, np.pi, THETA_BINS)),        # ABSOLUTE theta
                   float(_q(p.coherence, 0.0, 1.0, COHERENCE_BINS)),
                   float(round(float(p.scale), 3)), ...))
```

`core/slots.py:74` — `"ORIENTATION": ("theta", "elongation", "anchor_class")`: the first
slot of the orientation lenses is an image-frame angle. §6 forbids absolute position
being *scored*; an image-frame orientation in the identity is the rotational analogue and
is the reason M13 spent a step testing canonicalisation. M13 measured it as
**not decisive** (−0.011 / +0.097 / −0.027), so this is not the blocker — but the leak is
real and the canonicalisation that fixes it is dark (`PLACEMENT_FRAME_KEYPOINT`-style
flags only; no canonical-signature flag exists).

**Minimal fix.** Not worth doing before B1-1/B1-2: the class gate dominates it.

### B1-6 COSMETIC — the sub-pixel place map is computed twice per build

`eth80_frontend.py:399` and `:410` both call `pl.Pipeline._derive_anchor_positions(None, obs)`,
and that function itself calls `gabor_place_positions(observation, anchor_positions)`
(`orchestration/pipeline.py:1058-1063`) when the flag is on. With the flag on, one build
pays for it twice. No output difference (the function is pure).

---

## B2 — MATCHER

### B2-1 BLOCKS THE GOAL — placement cannot act where appearance is zero, and `q` *is* zero for a gated pair

`matching/alignment.py:630-645`

```python
if rel_weight is None:
    q = sim
elif skip_zero:
    q = [
        [(sim[i][j] * (1.0 - beta) + beta * rel_weight(i, j))
         if sim[i][j] > 0.0 else 0.0            # <- THE ZERO GATE
         for j in range(n_b)]
        for i in range(len(nodes_a))
    ]
else:
    q = [ ...identical expression... ]
```

**Goal requires** (§3): "Right place supports the mapping; wrong place opposes it. No
checkable edge = no credit and no penalty." **The code does**: a candidate whose node
similarity is 0 keeps weight 0.0 *whatever its placement agreement is*, so placement can
only reorder candidates that appearance has already made admissible. Combined with B1-2
(70.6 % of detected counterparts have similarity 0) the placement mechanism is
structurally unable to reach the population it is meant to decide.

**Minimal fix.** One line: drop the `if sim[i][j] > 0.0 else 0.0` guard so that
`q = sim·(1−β) + β·rel` can reach `β` for an appearance-zero pair. It is *not* sufficient
on its own (see B1-2: the cross-class signature carries no signal, AUC 0.512), which is
exactly why Part D does not put this first.

### B2-2 BLOCKS THE GOAL — "no checkable edge" is scored as a PENALTY (0.0), not neutral

`matching/alignment.py:536, 593-594` and `:1022-1031`

```python
    else:
        # [A1] no relations is NO relational support, not perfect agreement.
        agree_u = 0.0
...
    else:
        agree_v = 0.0
    return min(agree_u, agree_v)
```

```python
def _quality(u, v, mp, rv, beta):
    s = matrix[idx_a[u]][idx_b[v]]
    if s <= 0.0:
        return 0.0
    rel = _rel(u, v, mp, rv)
    return s * (1.0 - beta) + beta * rel        # rel = 0 for a degree-0 pair
```

With the shipped `MATCH_SYSTEMATICITY_WEIGHT = 0.5` (`parameters/_text.py:2418`), a pair
with no checkable relation scores **half** the quality of an otherwise identical pair
with perfect relational agreement. §3 says that is forbidden: no checkable edge must be
neutral (rel = 1.0, or the term dropped from that pair's blend), not zero.

**Measured (probe):** 43–100 % of the detected truth pairs have a degree-0 end
(42.9 %, 57.1 %, 66.7 %, 100 %, 50 %, 37.5 % over the six cells). So the penalty is
applied to a majority of exactly the pairs the goal asks us to find.

**Minimal fix.** `_relational_agreement` should return `None` (or 1.0) when *both* sides
are degree-0 and the blend should renormalise over the terms that exist. Medium risk:
`rel` is used in the assignment, the swap objective and the final prune, so it is one
change in three places (`_quality`, `_objective`, the prune loop) behind a flag.

### B2-3 DEGRADES — candidate generation is complete, but the optional prune is unrecoverable

`matching/alignment.py:853-870`

```python
if sim is None:
    sim = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for j, v in enumerate(nodes_b):
            sim[i][j] = float(sim_fn(u, v))
...
matrix = [
    [(0.0 if float(sim[i][j]) < unmatched_cost else float(sim[i][j]))
     for j in range(len(nodes_b))]
    for i in range(len(nodes_a))]
```

**Answer to "is the full matrix built, and is anything pruned before assignment":** the
full n×m matrix is always built (no candidate truncation anywhere), and at the shipped
`MATCH_UNMATCHED_COST = 0.0` (`parameters/_text.py:2353`) nothing is pruned. With
`MATCH_UNMATCHED_COST_DERIVED` on, the price is the mean positive similarity
(`joint_mcs.py:320-333`) and **any pair below the price is zeroed and can never be
recovered by placement** (B2-1 gate). So the prune is a latent version of the same
blocker.

### B2-4 DEGRADES — the assignment objective, and where the placement term enters it

Appearance/structure blend (assignment): `matching/alignment.py:597-620, 630-645`

```python
q(i, j) = sim(i, j)·(1 − β) + β·rel(i, j)
```
with `β = MATCH_SYSTEMATICITY_WEIGHT = 0.5` and `rel` = the conjunctive typed-relation
agreement of the pair's mapped neighbourhood (§B2-2).

Placement term (dark): `matching/placement_matcher.py:368-380`

```python
term = L.plus_score(u, int(vid), ctx, side_a.incident, side_a.index,
                    side_b.index, fit, pfit, a_e, a_g, tol=..., normalise=norm,
                    min_judged=_min_judged())
new = (1.0 - beta) * float(s) + beta * float(term)
```
with `beta = 0.5` from `matching/placement_frozen_params.json`, and `term` the frozen
LLR map in [0,1] centred at 0.5. It is applied to **every positive cell** of the row
(`enun_rows`, `placement_matcher.py:228-238`), so it can reorder candidates within a row
— but it can never *create* a candidate, because zero cells are not enumerated. **The
two β's are different parameters with the same name** (matcher blend vs in-row blend);
the ledger treats them separately.

### B2-5 DEGRADES — the refinement's move set and acceptance rule

Refinement: `matching/alignment.py:1155-1200` — re-assign on the same objective, then
`_swap_pass`:

```python
if _swap_delta(mp, u1, u2, v1, v2, beta) > _EPS:
    mp[u1] = v2
    mp[u2] = v1
```
Acceptance is a **strict gain** on the joint objective; a round is accepted only if it
improves (`if objective > current_objective + _EPS`), and the best mapping seen is
returned. Swaps are offered only along edges of either graph (`_swap_candidates`,
`:1225-1250`), so two isolated nodes can never exchange partners.

Final pruning (`:1215-1240`) is the fixed point:

```python
for u, v in sorted(filtered.items()):
    s = matrix[idx_a[u]][idx_b[v]]
    if s <= 0.0: continue
    if has_edges: rel = _rel(u, v, filtered, rev_filtered)
    else:         rel = 1.0
    q = s * (1.0 - final_beta) + final_beta * rel
    if q <= unmatched_cost: continue
```

**Can a correct pair be dropped here?** With `unmatched_cost = 0.0` only a zero-quality
pair is dropped, and a zero-quality pair has `s = 0` — already gated in B2-1. So the final
prune is *not* an independent loss channel at the defaults; it is the same gate.

### B2-6 COSMETIC — `skip_zero` is inert

The `elif skip_zero:` and `else:` branches of `_assign` (`alignment.py:633-645`) are
**character for character the same expression**, so the parameter saves no evaluation and
its docstring ("the evaluation is saved") is false. Fixing it is provably output-neutral
and is *listed, not fixed*, because it changes no behaviour and no number.

### B2-7 THE RESIDUE — every place a detected true pair can be lost, with the code

Probe (6 cells: 3 synthetic + 3 real × {exact shift, 5°}), truth = the tight rule, arm A
= shipped aligner, placement OFF. **39 detected truth pairs: 3 correct (7.7 %).**

| # | loss channel | code | observed |
|---|---|---|---|
| 1 | **lens-class gate** | `joint_mcs.py:307` `x[0] != y[0] → 0.0` | **29 / 39 (74 %)** |
| 2 | zero-similarity weight gate | `alignment.py:636-645` `if sim>0 else 0.0` | same population, at the assignment |
| 3 | assignment contest (partner taken by another node) | `alignment.py:648` `_max_weight_bipartite_matching(w)` | 6 / 39 |
| 4 | relational-zero penalty | `alignment.py:536/593/1022-1031` | halves quality, not a hard loss |
| 5 | final prune | `alignment.py:1228-1230` | **0 / 39** at the defaults |
| 6 | budget exhaustion stops the refinement | `alignment.py:1146-1150` (`pair_evals >= max_pair_evals`) | possible: `MATCH_ALIGN_MAX_PAIR_EVALS = 200000` is charged per `_rel` call, and one refinement matrix over ~600×600 nodes is 360 000 evaluations — the budget is exhausted **inside the first round**, so at most one refinement round runs and `_swap_pass` may not run at all |
| 7 | direction selection discards the other mapping | `joint_mcs.py:655-669` (keeps the better of the two directional solves, returns one mapping) | a pair found by the other direction is dropped wholesale |
| 8 | instrument | `Topology/placement_m12_read_real.py:_truth` (1 px + 2× separation + mutual) | not a matcher loss |

Channel 1 is three quarters of the residue, which is why Part D's recommendation is
upstream of the matcher.

---

## B3 — SCORER

### B3-1 The node term, the edge term, the placement term

`matching/joint_mcs.py:543-556`

```python
node_term=(nodes/node_denom) if node_denom else 0.0
edge_term=(edge_credit/edge_denom) if edge_denom else 0.0
...
w=float(P.MATCH_EDGE_WEIGHT)
if separate:
    score=2.0*((1.0-w)*node_term+w*edge_term)
else:
    score=2.0*(nodes+w*edge_credit)/denom if denom else 0.0
```

with (`:487-497`, `:522-527`)

```python
nodes=math.fsum(node_sim(u,v) * (isolated_w if (deg_a.get(u,0)==0 or deg_b.get(v,0)==0) else 1.0)
                for u,v in sorted(al.mapping.items()))
edge_credit=math.fsum(credits)
denom=len(na)+len(nb)+len(a.edges)+len(b.edges)
```

Placement term (`matching/joint_mcs.py:640-648`):

```python
ev, judged, share = placement_evidence(dict(al.mapping), _side_a, _side_b, _fit, _pfit,
                                       float(_b["_a_exist"]), float(_b["_a_geom"]))
w_score = float(getattr(P, "PLACEMENT_SCORE_WEIGHT", 0.0))
score = score_shipped + w_score * ev
```

and the evidence itself (`joint_mcs.py:110-158`):

```
E = Σ_{judged mapped pairs} (plus_score(pair) − 0.5) / (n_judged + k),   k = PLACEMENT_SCORE_SHRINK = 8
```

`plus_score` (`matching/placement_llr.py:360-365, 325-357`):

```
plus = clip(0.5 + (a_e·e_sum + a_g·g_sum) / (2·LLR_CLIP·SLOT_CAP), 0, 1)
e_sum = Σ_slots llr(slot) + mu_wrong·(cap − n)      [no padding under `normalise`]
```
with `a_e = 0.40625`, `a_g = 0.84375`, `LLR_CLIP = 3.0`, `SLOT_CAP = 8` (frozen JSON).

**Which of these can satisfy §4 ("reward satisfied placement, not edge count, node
count, or coverage")?** Only the third. `node_term` and `edge_term` are **sums over the
mapping divided by the graphs' total sizes** — i.e. coverage-weighted means: matching more
nodes, or preserving more edges, raises the score at equal per-pair quality. That is the
shipped MCS form, not a placement failure, but it *is* a §4 violation and it is why every
retrieval arm had to use a bespoke score (M11's note, `parameters/_text.py:1359-1385`).

### B3-2 Coverage neutrality of E

E's denominator is `n_judged + k`, so a candidate that maps 990 pairs and one that maps
800 are compared on **mean judged quality with an 8-pair shrinkage**, not on how many
pairs were judged. Within the term this *is* neutral (M10's coverage failure is what the
denominator exists to prevent). Its weakness is the opposite one: with `k = 8` and
`n_judged ≈ 105–206`, the shrink is only 4–7 %, so E is essentially an unshrunk mean, and
one bad pair moves it as much as one good pair.

### B3-3 CAN w·E FLIP A DECISION? — arithmetic

```
E   = Σ(plus − 0.5) / (n_judged + 8)          plus ∈ [0,1] ⇒ |plus − 0.5| ≤ 0.5
|E| ≤ 0.5 · n/(n+8)
```

| quantity | value | source |
|---|---|---|
| judged share actually observed | 0.12 (brief) / **0.258** (M12 read, mean) | `m12_read_real.json.frame_chain` |
| n_judged per pair | median **105**, mean 109, max 214 | `m13_read_real_progress.jsonl` (24 real cells) |
| theoretical bound at n=105 | |E| ≤ 0.5·105/113 = **0.464** | |
| **observed** E range | **[−0.0695, +0.0102]**, median **−0.0553** | same 24 cells |
| w = 0.375 ⇒ w·E | **[−0.026, +0.004]**, median −0.021 | |
| shipped score, truth − best rival | p25 **0.0027**, median **0.0228**, p75 0.0563 | 252-query bank |
| w·E, theoretical maximum | 0.375 × 0.464 = **0.174** | |

**Answer: yes, arithmetically — and it is the same size as the decision.** `|w·E|` at the
shipped weight (0.021) sits *on* the median margin (0.023) and above the lower quartile
(0.003), so it can flip any decision whose margin is below ~0.026 — a quarter of the bank
by construction. Measured on the 252-query bank by re-ranking:

| w | queries whose ranking changes | truth-first rate |
|---|---|---|
| 0 | — | 0.786 |
| **0.375 (shipped)** | **9 / 252 (3.6 %)** | **0.766 (−0.020)** |
| 1.0 | 19 / 252 | 0.742 |
| 3.0 | 42 / 252 | 0.690 |

So the term is **not** too small to matter; it is large enough to flip ~4 % of decisions
and it flips them **the wrong way**, because the *level* of E is systematically negative
(median −0.055 on real pairs: most judged pairs sit below the estimator's own 0.5 centre).
That is consistent with M13's read, where C − A was exactly 0.000 on the mapping: an
almost-constant negative offset does not reorder.

### B3-4 The weights: where each comes from, and which are at a boundary

| weight | value | source | fitted? | state |
|---|---|---|---|---|
| `PLACEMENT_SCORE_WEIGHT` (pair-score w) | **0.375** (deployed) / **0.0** (parameter default) | `matching/placement_score_params.json` | yes, M11 softmax MLE | **AT THE BOUNDARY**: `"ml_at_boundary": true`, `"w_ml": 3.0`, `"w_top1": 0.375` — the deployed number is a **tie-break by FIT top-1**, not an MLE |
| M13 margin refit | none | `m13_margin.json` | attempted | **AT THE BOUNDARY** (ŵ = 3.0, profile [2.75, 3.00]); shipped weight kept |
| `beta` (in-row placement blend) | 0.5 | `matching/placement_frozen_params.json` | M8 FIT | frozen, interior |
| `a_exist` / `a_geom` | 0.40625 / 0.84375 | same | M8 MLE on FIT grid | frozen (grid values — both are on the 0.0625 lattice) |
| `MATCH_EDGE_WEIGHT` (node/edge blend) | 0.5 | `parameters/_text.py:2425` | **no** | hand-set |
| `MATCH_SYSTEMATICITY_WEIGHT` (β) | 0.5 | `parameters/_text.py:2418` | **no** | hand-set |
| `MATCH_UNMATCHED_COST` | 0.0 | `parameters/_text.py:2353` | **no** | hand-set (= no price, no prune) |
| `PLACEMENT_SCORE_SHRINK` k | 8.0 | `parameters/_text.py:1361` | **no** | hand-set |

Note the deployed parameter default for w is **0.0**, not 0.375: the shipped pair score
contains no placement term unless a read loads the JSON. Every read that reports "P" has
therefore had to inject the weight — and no artifact records whether it did.

---

## C — THE CONFIGURATION LEDGER

### C1. Every `FE_* / PLACEMENT_* / MATCH_*` parameter in the tree (82)

`state` = what the **default** does: `dark` (off/zero), `default-ON`, or `ADOPTED`
(the comment says adopted). `introduced` is the first commit that mentions the name in
`parameters/_text.py` (this branch is a linear history from `b541c99`, so flags older than
the import all report the import commit and their mission tag comes from their own
comment).

| flag | default | what it changes | introduced (mission) | state |
|---|---|---|---|---|
| `FE_SUBPIXEL_GABOR_ANCHOR` | `False` | FE_SUBPIXEL_GABOR_ANCHOR: refine the Gabor pixel anchor to sub-pixel with a 2D quadratic fit of the channel response around the in | M5 `8b3e4aa` | dark |
| `FE_KEYPOINTS` | `False` | FE_KEYPOINTS: emit the true-keypoint frame layer (lenses/keypoints.py) -- Harris corners, multi-scale LoG blobs and arc-length bou | M5 `8b3e4aa` | dark |
| `FE_PLACE_LAYER` | `False` | FE_PLACE_LAYER: salt the pattern-node key with the detection's own anchor, so each detection of a place lens is its OWN node inste | M7 `fa8d325` | dark |
| `PLACEMENT_MATCHER_ENABLED` | `False` | THE CREDITED SCORER IN THE DEPLOYED MATCHER. Mission 8 credited P_plus_ctx -- existence + geometry with two fitted coefficients, i | M9 `68b5d53` | dark |
| `PLACEMENT_MATCHER_PRESENCE` | `False` | : The geometry coefficient set to ZERO, everything else identical | M9 `68b5d53` | dark |
| `PLACEMENT_MATCHER_STRICT` | `False` | : TEST MODE: a failure in the dark path RAISES instead of falling back, and a : missing side is an error rather than a silent skip | M9 `68b5d53` | dark |
| `PLACEMENT_SPACING_TOL` | `False` | : [MISSION 10 / step 4] Spacing-aware geometry | M10 `fb885ff` | dark |
| `PLACEMENT_SPACING_RADIUS` | `False` | — | M10 `fb885ff` | dark |
| `PLACEMENT_PAIR_NORMALISED` | `False` | : [MISSION 10 / step 4b] The placement term as a MEAN over the pairs it judged, : not a sum over the slot cap | M10 `794bcf6` | dark |
| `PLACEMENT_E_NORMALISED` | `False` | : [MISSION 12 / step 2] THE EVIDENCE ON ONE SCALE. ``PLACEMENT_PAIR_NORMALISED`` : removes the padding but leaves the denominator  | M12 `85ead16` | dark |
| `PLACEMENT_E_MIN_JUDGED` | `0` | — | M12 `85ead16` | dark |
| `PLACEMENT_FRAME_KEYPOINT` | `False` | : [MISSION 12 / step 3] THE FRAME SOURCE. The keypoint estimator (Gaussian : structure tensor at the keypoint layer's sigma ladder | M12 `5742c8c` | dark |
| `PLACEMENT_FRAME_FALLBACK` | `'none'` | : [MISSION 13 / step 1a] WHAT THE KEYPOINT PROVIDER FALLS BACK TO. M12 fell : back to the multiscale ladder, which is the WORST of | M13 `0a1a28b` | ADOPTED |
| `PLACEMENT_FRAME_KEYPOINT_SIGMA` | `'largest'` | : [MISSION 13 / step 1] WHICH KEYPOINT SIGMA WINS: "smallest" (the deployed rule: : the smallest sigma whose coherence clears the  | M13 `86573bb` | ADOPTED |
| `PLACEMENT_FRAME_SUBPIXEL` | `False` | : [MISSION 13 / step 1c] SUB-PIXEL FRAME SAMPLING for the field providers | M13 `0a1a28b` | dark |
| `PLACEMENT_SCORE_EVIDENCE` | `False` | : [MISSION 11 / step 1] THE PAIR SCORE ITSELF. Until now placement only moved : the MAPPING: `JointMCS.score` is computed after al | M11 `9e0e8ee` | dark |
| `PLACEMENT_SCORE_SHRINK` | `8.0` | — | M11 `9e0e8ee` | default-ON |
| `PLACEMENT_SCORE_WEIGHT` | `0.0` | : The fitted weight lives in matching/placement_score_params.json (provenance and : the FIT likelihood next to it); the environmen | M11 `9e0e8ee` | dark |
| `PLACEMENT_FRAME_MULTISCALE` | `False` | : [MISSION 11 / step 5] WHICH FRAME PROVIDER THE PAYLOADS ARE BUILT ON. DARK. : : `build_edges` has always used the SINGLE 7x7 str | M11 `a2c529f` | dark |
| `MATCH_GLOBAL_CONSISTENCY_ENABLED` | `False` | Use global-consistency alignment for correspondence in the live MCS path | ? `b541c99` | dark |
| `MATCH_UNMATCHED_COST` | `0.0` | Explicit price of leaving a node unmatched | ? `b541c99` | dark |
| `MATCH_UNMATCHED_COST_DERIVED` | `False` | Derive the rejection price from the objective's own scale: the arithmetic mean of POSITIVE node similarities over the same-lens ca | [PHASE 3 / F2] `b541c99` | dark |
| `MATCH_SIG_SIM_CACHE` | `False` | Content-keyed signature similarity cache, shared across pairs inside one process: `matching/sig_sim_cache.py` keys on the WHOLE si | M12 `df06ce4` | dark |
| `MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE` | `False` | Isolated frame nodes stay in the graph as context but earn no correspondence credit and are not paired | [PHASE 3 / F1] `b541c99` | dark |
| `MATCH_RELATIONAL_CONJUNCTION` | `False` | Relational conjunction at the correspondence: the final quality filter uses beta=1 and the pair weight is sim*(1-beta)+beta*rel | [PHASE 3 / FIX 1] `b541c99` | dark |
| `MATCH_WALKER_ALLOW_CLASS_MI` | `False` | May the CLASS-LABELLED mutual information enter the unsupervised correspondence? P.VOCAB_MI_TABLE holds, per relation type, the mu | ? `b541c99` | dark |
| `MATCH_ALIGN_BETA_ANNEAL` | `False` | Anneal beta 0 -> 1 over the refinement rounds: beta_k = (k+1)/(rounds-1), k=0..rounds-2; the final round and the final quality fil | [PHASE 3 / FIX 1] `b541c99` | dark |
| `MATCH_ALIGN_RELATIONAL_SEED` | `False` | Bias the seed by incident relation-type overlap so the pure node-similarity Hungarian cannot start from an arbitrary same-lens pai | [PHASE 3 / FIX 1] `b541c99` | dark |
| `MATCH_ALIGN_UNMAPPED_CREDIT` | `0.0` | Partial credit for an UNMAPPED partner | ? `b541c99` | dark |
| `MATCH_ISOLATED_NODE_WEIGHT` | `1.0` | Down-weight a node with no incident relation instead of removing it | ? `b541c99` | default-ON |
| `MATCH_EXCLUDE_ISOLATED_FLOOR` | `0.5` | Floor for MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE. The flag is a DOWN-WEIGHT, not a deletion: a zero weight makes an edgeless g | ? `b541c99` | default-ON |
| `MATCH_SYSTEMATICITY_WEIGHT` | `0.5` | Weight of the relational (systematicity) modulation of the alignment. | ? `b541c99` | default-ON |
| `MATCH_ALIGN_ROUNDS` | `4` | Bounded fixed-point rounds for the global alignment refinement. | ? `b541c99` | default-ON |
| `MATCH_ALIGN_MAX_PAIR_EVALS` | `200000` | Pair-evaluation budget for the refinement | ? `b541c99` | default-ON |
| `MATCH_EDGE_WEIGHT` | `0.5` | Weight of the relational term in the combined fuzzy score (the node term is 1 - this) | ? `b541c99` | default-ON |
| `MATCH_EDGE_MIN_CREDIT` | `False` | Edge-weight credit mode | ? `c308416` | dark |
| `MATCH_EDGE_DENOMINATOR_MODE` | `'all'` | How the pooled score charges edges it did not preserve | ? `b541c99` | default-ON |
| `MATCH_COVERAGE_EXPONENT_GAMMA` | `1.0` | Coverage exponent gamma applied to the edge mass | ? `b541c99` | default-ON |
| `MATCH_CATEGORICAL_AGREEMENT_GRADED` | `True` | When a presence-only relation has no measured graded axes, credit the categorical agreement by the FRACTION of identity axes that  | ? `b541c99` | default-ON |
| `MATCH_NULL_CALIBRATED` | `True` | Null-calibrated mass + floor-free conjunction in the live evidence path | ? `b541c99` | default-ON |
| `MATCH_MIN_CONSENSUS_SUPPORT` | `0.5` | Minimum support weight for a consensus edge to be kept (no invented edges). | ? `b541c99` | default-ON |
| `MATCH_SEARCH_MAX_EXPANSIONS` | `4096` | Best-first extension search budgets. | ? `b541c99` | default-ON |
| `MATCH_SEARCH_MAX_OPS` | `200000` | — | ? `b541c99` | default-ON |
| `MATCH_NEIGHBOR_BIN_CREDIT` | `False` | Two opt-in tolerance readings of the NODE similarity, applied AFTER the tree's own TypedSignature.similarity | [STEP F / TOLERANT GRADED CHANNELS] `b541c99` | dark |
| `MATCH_POPULATION_EMD` | `False` | MATCH_POPULATION_EMD: when both signatures carry non-empty population vectors of equal arity (the CONTENT lenses: CONTOUR_PEAK, Co | ? `b541c99` | dark |
| `MATCH_ASSIGNMENT_EXACT` | `True` | matching/structural_similarity.py _max_weight_bipartite_matching (the node correspondence of mcs_score, mcs_similarity and evidenc | ? `b541c99` | default-ON |
| `MATCH_SIGNATURE_MATRIX_BLOCKS` | `True` | memory/cores.py _mcs_accumulate (and the dark _global_correspondence): build the node-similarity matrix of the correspondence with | ? `b541c99` | default-ON |
| `MATCH_EDGE_MASS_CONTAINMENT` | `False` | THE NEW FLAGS. Every one of them defaults to the HISTORICAL behaviour, so the overlay loads without changing a single number; each | [PHASE 2] `b541c99` | dark |
| `MATCH_SKIP_INCOMPARABLE_RELATIONS` | `False` | — | ? `b541c99` | dark |
| `MATCH_KEEP_HIERARCHY_MEMBERS` | `False` | — | ? `b541c99` | dark |
| `MATCH_TEMPLATE_MEDIAN_EXEMPLAR` | `False` | — | ? `b541c99` | dark |
| `MATCH_CONSENSUS_COUNT_ABSTENTIONS` | `False` | — | ? `b541c99` | dark |
| `MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE` | `False` | THE NEW FLAGS. Both default to the HISTORICAL behaviour, so the series applies without changing a single number; each is an arm, s | [PHASE 3] `b541c99` | dark |
| `MATCH_SEPARATE_TERM_NORMALIZATION` | `False` | — | ? `b541c99` | dark |
| `PLACEMENT_PAYLOAD_ENABLED` | `False` | ============================================================================ [PLACEMENT] Correspondence decided by WHERE a node si | ? `29fa2d7` | dark |
| `PLACEMENT_SCORER_ENABLED` | `False` | Let placement support modify node similarity in assignment (Part 3). | ? `29fa2d7` | dark |
| `PLACEMENT_GATE_ENABLED` | `False` | Gate placement use on a TRUTH-FREE quantity (the similarity margin, Part 4). | ? `29fa2d7` | dark |
| `PLACEMENT_WEIGHT` | `0.25` | : ONE pre-stated weight; no sweep (Part 4) | ? `29fa2d7` | default-ON |
| `PLACEMENT_TOLERANCE` | `None` | : The frozen residual tolerance in scale units (Part 3) | ? `29fa2d7` | dark |
| `PLACEMENT_MAX_NEIGHBOUR_CREDIT` | `1.0` | : Cap on ONE neighbour's contribution to a node's support (Part 3). | ? `29fa2d7` | default-ON |
| `PLACEMENT_SUPPORT_SIGMA` | `1.0` | : Support is a bounded function of residual/tolerance | ? `29fa2d7` | default-ON |
| `PLACEMENT_COLLAPSE_PLACES` | `False` | THE PAYLOAD BUILDER'S OWN SWITCHES. Until now these six were read by `matching/placement_edges.build_edges` through `_param(name,  | M14 `` | dark |
| `PLACEMENT_PLACE_GRID_PX` | `1.0` | : The site grid of that collapse, in pixels. | ? `` | default-ON |
| `PLACEMENT_ENDPOINT_LENSES` | `()` | : Endpoint whitelist (Mission 7 step 2) | ? `` | dark |
| `PLACEMENT_EXCLUDE_LENSES` | `()` | : Endpoint blacklist (stage 2a (i)) | ? `` | dark |
| `PLACEMENT_RIDGE_ACROSS_ONLY` | `False` | : A placement edge must CROSS the ridge, not run along it (stage 2a (ii)). | ? `` | dark |
| `PLACEMENT_RIDGE_ACROSS_MIN` | `0.5` | : The across-ness that counts as "crossing" (cosine of the frame difference). | ? `` | default-ON |
| `MATCH_ALIGN_BUDGET_REFINEMENT_ONLY` | `True` | matching/alignment.py global_consistency_alignment: MATCH_ALIGN_MAX_PAIR_EVALS is the budget "for the refinement". The seed's simi | ? `b541c99` | default-ON |
| `MATCH_GLOBAL_ANCHOR_PRIOR` | `False` | memory/cores.py _global_correspondence: weigh the candidates of the global alignment by signature similarity x the anchor-position | ? `b541c99` | dark |
| `MATCH_RELATION_WALKER` | `False` | THE RELATION WALKER (matching/relation_walker.py): a correspondence grown along measured relations from a few seeds; signature sim | ? `b541c99` | dark |
| `FE_L3_SPINE` | `False` | : [L3] Content-token spine (mutual-best successor/predecessor + chain families). | ? `b541c99` | dark |
| `FE_L4_JUNCTIONS` | `False` | : [L4] Content-token junctions (vertices and occlusion T-points). | ? `b541c99` | dark |
| `FE_SUBPIXEL_PEAK_POS` | `False` | : [T4/PART1] Sub-pixel peak localisation | ? `c308416` | dark |
| `FE_L6_ADDRESS` | `False` | : [L6] Weisfeiler-Lehman structural address (k=1,2,3) on kept relation families. | ? `b541c99` | dark |
| `FE_SOFT_CHANNEL_GATE` | `True` | Soft gates -- replace hard pass/fail cut-offs with graded weights | ? `b541c99` | default-ON |
| `FE_SOFT_L3_L4_GATE` | `True` | : [1b] Soft L3/L4 gates: every hard cut (reach, mutual-best, corner turn, : T-point gap, dedup radius) becomes a graded weight; de | ? `b541c99` | default-ON |
| `FE_SOFT_GATE_PERCENTILE` | `0.95` | : [1c] Signal percentile used to normalize candidate signals (instead of the : maximum) | ? `b541c99` | default-ON |
| `FE_SOFT_GATE_BAND` | `0.1` | : [1c] Band half-width around the threshold (in normalized-signal units) | ? `b541c99` | default-ON |
| `FE_SOFT_GATE_NOISE_FLOOR` | `0.05` | : [1c] Absolute noise floor added to every graded weight | ? `b541c99` | default-ON |
| `FE_SOFT_GATE_EDGE_WEIGHT` | `True` | : [1a] When True, the graded weight also becomes the edge's signal weight : (not only the EdgeParams.confidence slot), so weight-a | ? `b541c99` | default-ON |
| `FE_SOFT_GATE_CENTRE_SHIFT` | `0.0` | : [1c] DEV-derived SHIFT of the soft-gate ramp centre (in normalized-signal : units) so the total weight over DEV scenes matches t | ? `c308416` | dark |
| `FE_VECTOR_ADDRESS` | `False` | Vector structural address (replaces exact-hash MATCH with cosine) | ? `b541c99` | dark |
### C2. Every reported result, against the flag vector that produced it

`RECOVERED` means the artifact states the value. `PARTIAL` means the artifact states some
of it. `NOT RECOVERABLE` means nothing in the tree says what the run's configuration was
and it cannot be reconstructed.

| read | artifact(s) | flag vector |
|---|---|---|
| M7 place layer (look 2) | `m7_dev5*.json`, `m7_prec_places.json`; `m7_look2.json` is **absent** | **NOT RECOVERABLE.** The report names `FE_PLACE_LAYER` and `PLACEMENT_ENDPOINT_LENSES` but never states a vector, and the result file that would have carried it does not exist. |
| M8 credited scorer (16 scenes) | `m8_read.json`, `m8_preregistration.json` | **PARTIAL.** `frozen_configuration` = {context: iters 5, margin 0.0, max_rivals 1, mutual False, rival_tol 0.02; endpoint_whitelist: 8 lenses; beta, a_e, a_g}. **Front-end graph switches: NOT RECOVERABLE.** |
| M9 irregular / out-of-plane | `m9_checks_ir.json`, `m9_checks_oop.json` | **NOT RECOVERABLE.** Only arm names and numbers; the report names `PLACEMENT_MATCHER_ENABLED` and `_PRESENCE` but no vector, no digest. |
| M10 out-of-plane read | `m10_prereg_oop.json`, `m10_oop_check.json` | **PARTIAL.** `flags = {PLACEMENT_MATCHER_ENABLED: True, PLACEMENT_MATCHER_PRESENCE: "per arm", PLACEMENT_MATCHER_STRICT: False}` + `frontend_digest = 79dfeff2…`. The digest pins the tree but is **not invertible** into a vector. |
| M11 ETH-80 and synthetic | `m11_prereg_eth.json`, `m11_prereg_synth.json`, `m11_eth80.json`, `m11_synth.json` | **NOT RECOVERABLE** beyond `weight.frozen_before_this_read: true`. The frame provider is not recorded; the report mentions `PLACEMENT_FRAME_MULTISCALE` only as the dark alternative it did **not** use. |
| M12 real-photo read | `m12_prereg_real.json`, `m12_read_real.json` | **PARTIAL, the best before M13.** `frozen` = {w 0.375 from `placement_score_params.json`, edge_beta 0.5, `PLACEMENT_E_NORMALISED`, frame provider = "the shipped field provider", target_k 16, near-tie rule {margin_max 0.05, top_sim 0.99}}. Graph switches and `PLACEMENT_MATCHER_ENABLED` are implied by arm names, never written. |
| M13 real-photo read | `m13_prereg_read.json`, `m13_read_real.json` | **RECOVERED except for the defaults it did not name.** Stated: `FE_SUBPIXEL_PEAK_POS=True`, `FE_SUBPIXEL_GABOR_ANCHOR=True`, arms A/B/C, w from the JSON, 4 strata, per-stratum MDE, verdict rule. Not stated: `PLACEMENT_FRAME_KEYPOINT` (it ran **off** — the field provider), the matcher flags beyond A/B/C, `PLACEMENT_E_MIN_JUDGED`. |

**Plain statement.** No read in this repository records a complete flag vector. The single
most complete is M13's, and even it leaves the frame provider to the default — which is
exactly the switch M13 had just spent a step changing.

### C3. Adopted improvements that have NEVER been read together

| adopted in | flag(s) | appeared in a read? |
|---|---|---|
| M13 step 1 | `PLACEMENT_FRAME_KEYPOINT=1`, `PLACEMENT_FRAME_KEYPOINT_SIGMA="largest"`, `PLACEMENT_FRAME_FALLBACK="none"` | **NEVER.** This is the adopted frame provider and **no read has ever used it**: M12 froze the field provider before the decision existed, and M13's own read left `PLACEMENT_FRAME_KEYPOINT` off (the field). |
| M13 step 3 | `FE_SUBPIXEL_PEAK_POS=1`, `FE_SUBPIXEL_GABOR_ANCHOR=1` | M13's read only — and never together with the adopted frame provider above. |
| M12 step 2 | `PLACEMENT_E_NORMALISED=1` | M12 and M13 — never with the adopted frame provider, and with `PLACEMENT_E_MIN_JUDGED = 0`. |
| M11 step 1 | `PLACEMENT_SCORE_EVIDENCE=1` + w = 0.375 | M12, M13 — with a w at the fit's boundary (B3-4). |
| M7 | `FE_PLACE_LAYER=1` | **NEVER** in any read in this tree (it is the fix for B1-1). |
| M6 | `FE_KEYPOINTS=1` | **NEVER** in a read (probe only, §B1-2). |

So the deployed stack has **two adopted mechanisms that have never been measured in a
read** (the frame provider, the place layer) and **two more that have only ever been
measured against the field provider** (sub-pixel anchors, E normalisation).

---

## D — THE VERDICT

### D1. The gaps that BLOCK THE GOAL, cheapest decisive test first

| # | blocker | why it blocks | cheapest decisive test |
|---|---|---|---|
| **1** | **Identity is a (lens, quantised-signature) class, not a place** (B1-1, B1-2) | 70.6 % of detected counterparts are `sim = 0` because the two views label one place with different lens classes; 44–56 % of places carry several nodes. §1 and §5 are both violated, and every downstream number is capped by it. | **No read.** On the cached 6 cells: merge co-located detections into one node with a lens set, then recount (a) places vs nodes, (b) truth pairs with a *comparable* descriptor. The cross-class slot AUC is already measured at **0.512** — the merge must be followed by a real descriptor, so the test is: does a frame-relative descriptor (the payload vectors) separate true from impostor places at AUC > 0.7? |
| **2** | **Placement is multiplicative on appearance** (B2-1) and **"no checkable edge" is a penalty** (B2-2) | §3's two rules are both inverted: right place cannot rescue `sim = 0`, and no edge costs half the quality. | **No read.** One-line change: drop the `sim>0` guard; return `None` for a both-sides-degree-0 pair and renormalise. Measure arm A's geometric correspondence on the same 12 cells (M13 harness already does this). |
| **3** | **The pair score rewards coverage** (B3-1) | §4 forbids rewarding node/edge count; `2(nodes + w·edge_credit)/denom` is a coverage-weighted mean. | **No read.** On the 252-query bank: correlate candidate score with `|mapping|` at fixed per-pair quality — M10 already measured this for the placement term (606–921 mapped pairs, flat quality); the same test has never been applied to the shipped pair score. |
| **4** | **The anchor is nearest-pixel / lattice-quantised at the defaults** (B1-3, B1-4) | Detection 0.129 at 5° against 0.469 under a shift; the sub-pixel fix is dark and unread. | **No read.** M13's `placement_m13_repeat.py` already is the test; the missing arm is `FE_SUBPIXEL_*` **with** `PLACEMENT_FRAME_KEYPOINT=1`. |

### D2. Which of the thirteen missions' conclusions are now unsafe

| mission | conclusion | why it is unsafe |
|---|---|---|
| M4/M5/M6 (localisation) | sub-pixel anchors / keypoints improve localisation | measured **with the writers dark**: the anchor a node keeps is the one first written, and the sub-pixel place is only carried by `_place_pos` (B1-3). The localisation numbers are real but are not what the deployed graph does. |
| M7 (place layer) | one node per place | the layer is dark in every read (C3); the *deployed* graph still has 668 nodes at 373 places (B1-1). The conclusion stands as a measurement of a dark arm, not of the system. |
| M8 ("CREDITED") | the credited scorer beats the shipped one (0.387 vs 0.318) | taken with `PLACEMENT_MATCHER_ENABLED=1` on graphs whose identity is class-based, so it can only speak about the ~29 % of counterparts that are appearance-visible (B1-2). Its own vector is only PARTIAL (C2). |
| M9 irregular/OOP | the deployed matcher's credited arm is real | same cap; and the vector is **NOT RECOVERABLE** (C2). |
| M10 (coverage) | the placement term is a coverage statistic and the mean reading fixes it | that diagnosis is **sound and confirmed here** (B3-3), but the read ran against the field provider and the flag it validates (`PLACEMENT_PAIR_NORMALISED`) was never deployed. |
| M11 (retrieval) | placement-aware scoring reaches ETH-80 top-1 0.380 | ETH-80 is absent from this sandbox (M13 step 6: 25 directories, **0 files**), so the numbers cannot be re-derived here, and the vector is not recorded (C2). Treat as unverified in this tree. |
| M12 (real read) | P − N = −0.002 [−0.012, +0.007] | run with the **field** frame provider and the **smallest**-σ rule — the configuration M13 then measured as the *wrong* one (0.563 against 0.811). The interval answers a question about a superseded provider. Also pooled across four transforms (§M13's reason for re-registering). |
| M13 step 1 | the keypoint provider at σ=4 is adopted | adopted on a **node-level** measurement only; never in a read (C3). |
| M13 step 5 | C − A = 0.000 on the exact shift ⇒ placement's evidence does nothing | the bank was 8–98 nodes per stratum (MDE 0.20–0.70) and the provider was the field. A structural null at the shipped weight is still informative, but it is not a null about the mechanism. |
| **all "placement does not help" statements** | — | unsafe as a verdict on the mechanism, because the mechanism is only reachable for 29 % of the counterparts the front end detects (B1-2, B2-7). |

### D3. The single next change with the largest expected effect

**Give a place a descriptor that is comparable across lenses and stable across views, and
make that descriptor the node's identity — with the lens as evidence, not as a gate.**

Why this and not the matcher changes: the residue is 74 % class-gate
(29/39 pairs), and the gate cannot simply be removed, because the two classes' signatures
are *not the same measurement in different bins* — cross-class slot agreement for true
pairs is **0.080** against **0.127** for random pairs (**AUC 0.512**, i.e. nothing). Only
a descriptor that is defined relative to the place's own frame can be compared across
lenses; the placement payload already computes exactly that geometry, and M13 measured
that the frame it needs has an equivariance of 0.811 when the adopted provider is used.

Expected effect: it raises the population on which placement can act from **29 %** of
detected counterparts to the ~100 % the problem allows, which is the only change that
moves the ceiling that every mission since M7 has been measuring against.

**Cheapest decisive test (no read):** on the 6 cached cells, take the 48 class-gated truth
pairs, build for each place a small descriptor = the multiset of frame-relative
(angle, distance) vectors to its 4 nearest places plus the local coherence, and measure
true-vs-impostor AUC. If AUC ≥ 0.7 the descriptor is real and B1-1/B1-2 can be repaired
behind one flag; if it is near 0.5 the front end is the binding constraint and the mission
after this one must be a front-end mission, not a matcher mission.

---

## The one fix committed by this audit

**Bug (wiring, plainly broken, found by the ledger's own completeness check).** Six
switches that decide the *payload* — and therefore the graph's endpoint set — were read by
`matching/placement_edges.build_edges` through `_param(name, default)` and were **declared
nowhere**:

```
PLACEMENT_COLLAPSE_PLACES, PLACEMENT_PLACE_GRID_PX, PLACEMENT_ENDPOINT_LENSES,
PLACEMENT_EXCLUDE_LENSES, PLACEMENT_RIDGE_ACROSS_ONLY, PLACEMENT_RIDGE_ACROSS_MIN
```

Consequences, both real: `getattr(P, name, default)` always fell through, so no run config
or `PERCEPT_SET` override could set them; and because they were absent from
`eth80_frontend._FRONTEND_SWITCHES`, `Topology/placement_cache.key_for` **did not move
when they changed** — the identical defect Mission 8 found for `FE_PLACE_LAYER` ("a step-5
run with the flags OFF was served the flags-ON graphs"), still open on the payload path.
`PLACEMENT_COLLAPSE_PLACES` changes *which nodes are endpoints*, so this was a
graph-level hole. The five numeric settings of the two soft-gate switches that were
already in the digest (`FE_SOFT_GATE_PERCENTILE`, `_BAND`, `_NOISE_FLOOR`,
`_CENTRE_SHIFT`, `_EDGE_WEIGHT`) had the same shape of gap: the on/off switch keyed the
cache, the threshold did not.

**Fix.** Declared the six in `parameters/_text.py` with **exactly the fallbacks the
builder already used** (`False`, `1.0`, `()`, `()`, `False`, `0.5`), added all eleven
names to `_FRONTEND_SWITCHES`, and added `tests/test_m14_switch_digest.py`, which fails if
a build-path module ever reads a `FE_*`/`PLACEMENT_*` switch that is not in the digest
(it parses the tuple with `ast`, because a regex on `")"` stops at the tuple's own
comments).

**Proof the defaults did not move:** `Topology/placement_m12_perf.py --check` against the
Mission-13 baseline, with the cache cold (the new keys forced a rebuild — wall 171.9 s
against 69.7 s cached):

```
identical: True  (0 diffs)          # 14 pairs × 2 arms × {score, shipped, ev, judged, mapping}
per-pair A 4.69s  C 5.23s
```

**Suites:** `tests/` 19 tests OK, `Topology/test_placement.py` 17 OK,
`Topology/test_placement_m3.py` 12 OK.
