# Mission 7 — the place layer, and the powered DEV-5 test

Branch `arena/c316ca3a-percept6`, from `98c8849`. Commits: `fa8d325` (place
layer), plus the step 2-3 code and this report.

---

## 1. The place layer — built, dark, tested both ways

`core/place_key.py` salts the pattern-node key with the detection's **own
anchor**: `(u, it, sf, kind)` for the tiling lenses, `("kp", kind, idx)` for
keypoints. **No grid rounding anywhere** — the anchor the lens emitted is the
key. The stored `TypedSignature` stays position-free, so matching is unchanged.

The salt is applied at every MINT site (binder 1083/1574/1762, `lenses/v2`,
`lenses/v4`, `concept_graph`, `memory/builder`) **and, through `sig_hash_of(det)`,
at every LOOKUP site** that builds a `(lens, hash)` pair. That second part was
not optional: salting only the mints made the graph **lose** its
Orientation/Texture/Color/Boundary nodes entirely (657 → 496), because every
presence set and onset table was still looking them up unsalted.

| check | result |
|---|---|
| flag off → byte-identical | **PASS** — 657 nodes / 1167 edges, `4707831d1393797f`, unchanged after every edit in this step |
| place lenses 1:1 with detections | **PASS** — Texture 136 = 136 kept, Boundary 128 = 128, Color 102 = 102, Orientation 45 = 45, Contour 6 = 6 |
| deterministic, fresh processes | **PASS** — `ecf47c8eaaae52f7` twice |
| every place node has a position | **PASS** — 45/45 Orientation (27 before; the simple Orientation path emitted tuple anchors the step map cannot hold) |

Node count **657 → 913**, edges 1167 → 1249, build **9.8 s → 9.8 s** (no
measurable cost). All three flags together: 1249 nodes, 1310 edges, 10.2 s, with
Blob 208/208, BoundaryArc 190/190 and CornerPoint 7/7 fully distinct.

**Where distinctness still fails**, stated rather than hidden: Orientation 30/45
(the 15 are segment anchors sharing one region centroid — the observation does
not expose a per-segment arc position), Texture 105/121 and Color 75/83 (cells
whose argmax is the same pixel are one physical place counted twice). That is
the same co-siting phenomenon Mission 5 measured from the other side, and it is
now visible per node instead of being folded away.

## 2. Placement rewired on places

`build_edges` gained an `endpoint_lenses` **whitelist**: only a place can be a
placement endpoint; attribute and ridge nodes attach to the nearest place and
inherit its support. Default empty = shipped behaviour, so nothing changed
silently.

Coverage on the AMBIGUOUS population, restated on the new graph (keypoints +
place layer, 6 FIT scenes, radius 4.0 px stated before measuring):

| | Mission 6 | Mission 7 |
|---|---|---|
| nodes measured | 961 | 1141 |
| coverage @4 px | 0.736 | **0.735** |
| median distance | 2.00 px | 2.13 px |

Coverage holds at 0.735 while the population grew 19%: making each detection its
own node did not scatter the attribute nodes away from keypoint places.

Node-to-node precision on the place graph (6 FIT scenes, same truth as before):

| view step | repeat | loc median | loc p90 | identity |
|---|---|---|---|---|
| w1_rot | 0.297 | 1.278 | 2.737 | 0.644 |
| w2_translate | 0.409 | 1.119 | 2.644 | 0.603 |
| w3_scale | 0.417 | 1.523 | 2.706 | 0.721 |

Against the pre-place graph (M6 writer-on: repeat 0.371 mean, loc 1.140,
identity 0.528): **identity rises 0.528 → 0.656 mean, +0.13**, at essentially
unchanged localisation (1.140 → 1.307) and repeatability (0.371 → 0.374). That
is the expected trade of one node per place: more nodes to get right, and
distinctly fewer wrong identities.

## 3. The powered DEV-5 run

32 scenes (8 per family: SPHERE, BOX, CYLINDER, OCCLUSION) × 4 view steps,
**split by scene into FIT 16 / CHECK 16 before anything was fitted**, stratified
by family. LLR and presence densities refitted on FIT under the new graph
(8174 true / 14327 wrong residuals, μ_wrong −0.1737). 160 graphs, 28 min build.

### CHECK look 1 — everything may be an endpoint (the shipped rule)

| arm | mean | | arm | mean |
|---|---|---|---|---|
| N | 0.3019 | | P_presence | **0.3689** |
| R | 0.2372 | | P_shuffled | 0.2555 |
| R_fixed | 0.3234 | | W | 0.2976 |
| P | 0.3258 | | P_oracle | 0.3445 |

| contrast | effect | MDE | verdict |
|---|---|---|---|
| P − N | +0.0309 | 0.0254 | POSITIVE |
| **P − P_presence** | **−0.0422** | 0.0166 | **NEGATIVE** |
| P − P_shuffled | +0.0833 | 0.0397 | POSITIVE |
| R_fixed − R | +0.0888 | 0.0255 | POSITIVE |

Geometry beat appearance but lost to presence, in every family. Per the decision
rule, not credited.

### Step 4 repair (chosen on FIT, free) and CHECK look 2

The repair is a **configuration, not a weight**: endpoints restricted to place
lenses. FIT, same 16 scenes:

| endpoints | N | P | P_presence | P − P_presence | MDE |
|---|---|---|---|---|---|
| all | 0.3153 | 0.3522 | 0.3821 | −0.0295 | 0.0180 |
| **places** | 0.3153 | **0.3595** | 0.3522 | **+0.0070** | 0.0113 |

CHECK look 2 of 5 (correction 0.05/look → 0.10 cumulative), same graph, same
fit, places only:

| arm | mean | | arm | mean |
|---|---|---|---|---|
| N | 0.3019 | | P_presence | 0.3423 |
| R_fixed | 0.3234 | | P_shuffled | 0.2954 |
| **P** | **0.3502** | | W | 0.3021 |
| | | | P_oracle | 0.3840 |

| contrast | effect | scene SE | MDE | CI | p | verdict |
|---|---|---|---|---|---|---|
| P − N | +0.0543 | 0.0070 | 0.0197 | [+0.0405,+0.0681] | 0.0000 | POSITIVE |
| **P − P_presence** | **+0.0094** | 0.0040 | 0.0113 | [+0.0014,+0.0173] | 0.0260 | **INCONCLUSIVE** |
| P − P_shuffled | +0.0626 | 0.0100 | 0.0280 | [+0.0430,+0.0822] | 0.0000 | POSITIVE |
| P − W | +0.0550 | 0.0076 | 0.0213 | [+0.0401,+0.0699] | 0.0000 | POSITIVE |
| P − R_fixed | +0.0304 | 0.0071 | 0.0198 | [+0.0166,+0.0443] | 0.0001 | POSITIVE |
| P_oracle − P | +0.0376 | 0.0063 | 0.0176 | [+0.0253,+0.0498] | 0.0000 | POSITIVE |
| P_oracle − P_presence | +0.0469 | 0.0073 | 0.0203 | [+0.0327,+0.0612] | 0.0000 | POSITIVE |

**Per family, P now beats P_presence in all four:**

| family | n | N | P | P_presence |
|---|---|---|---|---|
| BOX | 1350 | 0.4904 | **0.5370** | 0.5252 |
| CYLINDER | 1856 | 0.4062 | **0.4359** | 0.4348 |
| OCCLUSION | 2224 | 0.2271 | **0.2545** | 0.2473 |
| SPHERE | 2847 | 0.3614 | **0.3860** | 0.3829 |

### The verdict

**No — geometry is not credited with beating presence.** The effect is
**+0.0094 with MDE 0.0113**: the confidence interval excludes zero
([+0.0014,+0.0173], p = 0.026) and the sign is positive in every family, but the
pre-registered decision rule requires the effect to exceed the MDE, and it does
not. It is a genuine near-tie leaning towards geometry, not a win, and not a
null either.

**Which change did it**: restricting placement endpoints to **places** — the
place layer plus the endpoint whitelist. Nothing else was touched: same scenes,
same split, same fit procedure, same arms, one graph throughout. The two
Mission-3-era defects that made every earlier geometry arm lose
(signature-class nodes standing in for places, and attribute/ridge nodes counted
as endpoints) are both gone.

## 4. Remaining levers, with their measured error

1. **The MDE is the binding constraint, not the effect.** +0.0094 against a
   0.0113 MDE with 16 CHECK scenes: the effect is ~0.83 of an MDE. More CHECK
   scenes is the only way to decide it, and the mission caps CHECK looks at 5
   (2 spent). Sizing for MDE 0.005 would need roughly 4× the CHECK scenes.
2. **Presence is a very strong arm on synthetic scenes.** It only asks whether a
   class exists, and on two-object scenes with a checker texture that is almost
   always yes; P_presence scores 0.3423 while P scores 0.3502. The margin is
   small because the ceiling is high for both.
3. **OCCLUSION is where geometry is weakest** (P 0.2545): partly occluded
   objects lose keypoints by construction, which is the coverage lever, not a
   scoring lever.
4. **Distinctness is still imperfect** — 15 of 45 Orientation nodes share a
   segment centroid because the observation exposes no per-segment arc position.
   That is a front-end gap with a known fix (expose segment arc length), and it
   is the largest remaining position error in the place population.

## 5. State

* Suites 17 + 17 + 12 OK (unchanged); every new flag dark:
  `FE_PLACE_LAYER=False`, `PLACE_LAYER_LENSES` declared,
  `PLACEMENT_ENDPOINT_LENSES=()`.
* Defaults byte-identical: `4707831d1393797f`, re-verified after the place layer.
* CHECK looks spent: **2 of 5** (correction 0.10 cumulative).
* Cache: pixels + switches + the 84-file source closure; 160 DEV-5 graphs warm
  under `FE_PLACE_LAYER=1`.

## 6. EVAL

Not run. The DEV gate requires P − P_presence beyond MDE; it is +0.0094 against
0.0113. EVAL needs a pre-registration committed before any EVAL graph exists,
one read, and power sized from CHECK — which first requires deciding this
near-tie on more CHECK scenes. Running EVAL now would spend it on an
undecided effect.
