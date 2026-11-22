# Mission 3 — Does placement GEOMETRY carry the gain?

**Verdict: NO.** The gain is carried by a residual-free *presence* term, not by
the geometry. The mission's prescribed diagnostic says the bottleneck is the
**payload and the frames, not the context** — and the second geometric
representation tried (frame-free angle/ratio relations) is null too, which
sharpens that to a payload-precision statement.

---

## Stage 0 — the baseline, restated honestly

EVAL v1 clustered on `family|scene|view`: **24 clusters from 4 scenes × 6 views**.
Views of one scene share its graph and differ only by a pose, so they are not
independent. Re-analysed with **SCENE** on top, using an exact two-sided
sign-flip permutation (4 scenes → 2⁴ = 16 assignments, so the exact p is floored
at 0.125):

| contrast | v1 (24 view clusters) | scene mean | exact p |
|---|---|---|---|
| P − N | +0.0612 [+0.0221, +0.1026] | +0.0597 | 0.125 (floor) |
| P − P_shuffled | +0.0301 [+0.0010, +0.0610] | +0.0186 | 0.375 |
| P − W | +0.0913 | +0.1046 | 0.125 |
| R − N | −0.0924 | −0.1046 | 0.125 |
| R_fixed − R | +0.0999 | +0.1156 | 0.125 |

**No contrast on EVAL v1 could have reached 0.05 at the scene level, whatever
its size.** The mission-2 `POSITIVE` for P − N is **withdrawn as a significance
claim**. The arms run reproduces `arms_eval.json` bit-for-bit
(`sha256 9d421df10728d0bc`), so this is the same data, not a second read.

By family, P − N on **RESOLVABLE** (the only family with a unique correct
answer) was **+0.010**; nearly the whole gain came from NEGATIVE-A/B and
CONTROL, where the payload provably cannot see the answer. That was the tell.

---

## What was built

**Lever 1 — a calibrated scorer** (`matching/placement_llr.py`). One estimator,
fixed in advance:

```
score(u→v) = Σ_{slots ≤ 8} clip(LLR(slot), ±3) / (2·3·8) + 0.5
```

* LLR from residual densities fitted on FIT, **true pairs vs the wrong RIVALS**
  (the near-tie candidates the assignment actually sees), not random nodes;
* **cap** 8 slots per candidate, canonical order;
* **pad** every uncheckable slot at `mu_wrong`, so a wrong candidate's expected
  score does not grow with its checkable-edge count;
* **clip** each slot so one wild residual cannot decide a node.

**`P_presence`** — the same caps, padding and normalisation with the residual
switched off: each slot is credited by the fitted per-class log-ratio of the
corresponding image edge *existing* under a true versus a wrong candidate.

**DEV-3, sized in SCENES** (`Topology/placement_dev3.py`): 40 scenes, 20 FIT /
20 CHECK, split by scene before any fitting, stratified by family, 3 views each,
every scene a fresh seed with its own shapes, materials, texture frequencies,
sizes and separations, poses inside a fixed envelope. Stage 0's scene-level sd
of 0.038 needs 12.6 scenes per arm for MDE 0.03; 20 CHECK scenes delivers
**MDE 0.014–0.022**, comfortably below the planning effect.

**Truth-free at match time, asserted.** `test_the_builder_needs_no_object_membership`
proves the deployed edge set is identical with and without object membership —
object ids only gate the NEGATIVE-A control. The scorer never reads truth; truth
labels the fitted pools and scores the result.

---

## CHECK — 20 scenes, n = 2860 near-tie ambiguous nodes, scene-clustered

FIT is well formed: 5847 true residuals, 14610 wrong, **wrong-pair mean −0.1174
< 0 < true-pair mean +0.1253**.

| arm | accuracy | | arm | accuracy |
|---|---|---|---|---|
| N | 0.3080 | | P | 0.3266 |
| R (frozen) | 0.2521 | | P_presence | **0.3542** |
| R_fixed | 0.3308 | | P_shuffled | 0.2874 |
| W | 0.3014 | | P_oracle (DEV-only) | 0.3325 |
| | | | P_oracle_presence | 0.3874 |

| contrast | effect | MDE | 95 % CI | p | verdict |
|---|---|---|---|---|---|
| **P − P_presence** | **−0.0229** | 0.0140 | [−0.0327, −0.0131] | 0.0003 | **NEGATIVE** |
| P − N | +0.0175 | 0.0153 | [+0.0068, +0.0282] | 0.0034 | POSITIVE |
| P − P_shuffled | +0.0360 | 0.0218 | [+0.0207, +0.0513] | 0.0001 | POSITIVE |
| P − W | +0.0231 | 0.0165 | [+0.0115, +0.0347] | 0.0008 | POSITIVE |
| P_presence − N | **+0.0404** | 0.0216 | [+0.0253, +0.0555] | 0.0001 | **POSITIVE** |
| P − R_fixed | −0.0054 | 0.0176 | [−0.0177, +0.0069] | 0.3993 | INCONCLUSIVE |
| R − N | −0.0515 | 0.0220 | [−0.0669, −0.0361] | 0.0000 | NEGATIVE |
| R_fixed − R | +0.0743 | 0.0257 | [+0.0564, +0.0923] | 0.0000 | POSITIVE |

Unambiguous bucket unharmed: P − N = +0.0022, MDE 0.0044, INCONCLUSIVE.
Strict label: +0.0052, INCONCLUSIVE.

**The mission's decision rule — "P − P_presence AND P − N both positive beyond
MDE" — fails on the first conjunct, decisively, in the wrong direction.**

---

## The diagnostic the mission prescribes

The failure tree says: *if geometry adds nothing over P_presence, run the oracle
arms; if oracle helps, work context; if not, work payload and frames.* Done.

The oracle arm replaces N with **renderer truth** as the neighbour mapping. It
violates truth-free-at-match-time by construction, is DEV-only, and can never
enter a deployed arm.

| contrast | effect | MDE | p | verdict |
|---|---|---|---|---|
| **P_oracle − P** | **+0.0033** | 0.0114 | 0.4280 | **INCONCLUSIVE** |
| P_oracle − P_presence | −0.0196 | 0.0153 | 0.0017 | NEGATIVE |
| **P_oracle_presence − P_presence** | **+0.0281** | 0.0138 | 0.0001 | **POSITIVE** |
| P_oracle − P_oracle_presence | −0.0476 | 0.0187 | 0.0000 | NEGATIVE |

Perfect context moves **presence** a long way (+0.0281) and moves **geometry**
by nothing (+0.0033). Even with the neighbours known exactly, geometry sits
0.0476 *below* presence.

**So the context is not the limit.** Mission 2's suspicion (a) — "context comes
from N's mapping and N is right only ~34 % of the time" — is **refuted as the
binding constraint**. The bottleneck is the payload and the frames: the
continuous residual is a noisier read of the same fact that the discrete
existence of the corresponding edge already encodes.

---

## Instrument defects found and repaired (old numbers kept)

1. **`R_fixed` was not a geometry arm at all.** It incremented its agreement
   count as soon as a counterpart *edge existed* and never read the parameters —
   so it was presence with a different denominator, which is why mission 2 found
   R_fixed "only matched N", and why the Stage-0 family table showed no
   geometric gain. It now compares the shipped `EdgeParams` (same class, same
   sign, every quantized axis within one bin), and separates immediately:
   R_fixed − R = **+0.0743**.
2. **`R_fixed`'s q-side lookup used the wrong index** (placement payloads for
   shipped edges), so every lookup missed and every candidate got the same
   credit — R_fixed stayed numerically identical to N for a second run. Fixed.
3. **The oracle arms were computed but not recorded** (the per-node record
   stored only the base arm tuple). Fixed.
4. **`fit_presence` fitted each class's *share* of existing edges**, which is
   nearly constant across candidates and made `P_presence` exactly N
   (measured: 0.0000). It now fits the rate of existence *given* the class,
   which is what varies.

New canaries in `Topology/test_placement_m3.py` (10 tests, all passing) exist to
catch exactly these: R's credit must vary across candidates, R_fixed must
separate an agreeing from a mismatched relation, P must separate a good
candidate from a bad one, padding must be `mu_wrong` and must remove the
checkable-edge-count advantage.

---

## Where this leaves the mission

**The gain is not attributable to geometry.** It is attributable to structure
(the corresponding edge existing), which the `P_presence` arm isolates and
which is deployed-truth-free.

The lever that remains, by the mission's own ordering, is **lever 3
(frame-free relations: angle at u between two neighbours, distance ratio)** and
**lever 4 (payload precision)** — the payload is what the oracle diagnostic
pointed at.

### Lever 3 was then built and is also null (CHECK look 4)

Frame-free relations: the unsigned angle at u between two neighbours and the
ratio of their distances. No orientation frame is needed, so this is the one
geometric read that does not depend on the field provider; it is invariant to
translation (only position differences enter) and to scale (a ratio of
distances). Mirror images are not distinguished, which is stated rather than
hidden. Capped at 6 neighbour pairs per candidate, same padded/clipped/summed
estimator, densities fitted on FIT.

| contrast | effect | MDE | p | verdict |
|---|---|---|---|---|
| **P_ratio − N** | **−0.0042** | 0.0089 | 0.2031 | **INCONCLUSIVE** |
| P_ratio − P_presence | −0.0446 | 0.0227 | 0.0000 | NEGATIVE |
| P_ratio − P | −0.0217 | 0.0177 | 0.0009 | NEGATIVE |

**Frame-free geometry does not even beat plain similarity.** P_ratio 0.3045
against N 0.3080. So the failure is not the choice of geometric *representation*
— an orientation-frame vector and a frame-free angle/ratio relation both fall
short of the discrete fact that the corresponding edge exists. That is a
payload-precision statement (lever 4), not a representation statement.

**Not delivered, and stated rather than hidden:**

* **Lever 4 (payload precision — refined anchor positions, per-edge tolerance
  scaled by coherence/span/pose) was not built.** It is the lever the evidence
  now points at, and it requires touching the front end's anchor quantisation
  rather than the scorer.
* **EVAL-2 was not run.** It requires a fresh scene set whose graph builds cost
  ~1 min each; at the scene count needed for MDE < the planning effect that is
  60–100 minutes of build time, which this session did not have. No EVAL-2 read
  exists, no EVAL-2 number is claimed, and the pre-registration was not
  committed because it would have had to be sized from a CHECK result that is a
  *null*. The DEV CHECK result above is decisive on its own terms (MDE 0.014
  against an effect of −0.0229) and is not a confirmation of anything.

**Looks used on CHECK: 4 of 5.** Logged: look 1 (full arms, defective
R_fixed), look 2 (oracle wiring, R_fixed still defective, no oracle recorded),
look 3 (R_fixed repaired, oracle recorded), look 4 (lever 3, frame-free
relations). With 5 looks the corrected threshold is 0.05 / 4 = 0.0125; every
decisive contrast above has p ≤ 0.0034, well inside that.
