# Stage 0 — EVAL v1 re-analysed with SCENE as the top-level cluster

An **analysis correction on spent data**, not a new read. The arms run was
re-executed only to emit per-node records; it reproduces the published table
bit-for-bit (`arms_eval.json` sha256 `9d421df10728d0bc` before and after), so
the numbers below are the same data seen honestly.

## How EVAL v1 was clustered

`family|scene|view` — **24 clusters**, from **4 scenes × 6 views**. Views of one
scene share its geometry, materials and node layout and differ only by a pose
applied to the same graph, so they are not independent. The interval was
therefore far narrower than the evidence supports.

## Re-analysis at scene level

Top-level unit = **scene**, of which EVAL v1 has **four**. With four clusters a
bootstrap says nothing useful, so the test is an **exact two-sided sign-flip
permutation** over the 2⁴ = 16 sign assignments of the four per-scene paired
mean differences.

| contrast | v1 (24 view clusters) | scene mean | per-scene (control / neg_a / neg_b / resolvable) | exact p | floor |
|---|---|---|---|---|---|
| P − N | +0.0612 [+0.0221, +0.1026] | +0.0597 | +0.050 / +0.090 / +0.090 / +0.010 | 0.125 | 0.125 |
| P − P_shuffled | +0.0301 [+0.0010, +0.0610] | +0.0186 | −0.025 / +0.048 / +0.045 / +0.006 | 0.375 | 0.125 |
| P − W | +0.0913 [+0.0450, +0.1387] | +0.1046 | +0.150 / +0.107 / +0.107 / +0.055 | 0.125 | 0.125 |
| R − N | −0.0924 [−0.1223, −0.0650] | −0.1046 | −0.150 / −0.083 / −0.083 / −0.103 | 0.125 | 0.125 |
| R_fixed − R | +0.0999 [+0.0688, +0.1357] | +0.1156 | +0.175 / +0.079 / +0.079 / +0.129 | 0.125 | 0.125 |

**The exact two-sided p cannot go below 2/16 = 0.125 with four scenes.** No
contrast on EVAL v1 could have reached 0.05 at the scene level, whatever its
effect size. The `POSITIVE` verdict reported for P − N was a view-level
artefact of counting six correlated views of the same scene as six independent
pieces of evidence, and it is withdrawn as a significance claim.

## What the corrected numbers say

1. **P − N is positive in all four scenes** (+0.010 to +0.090) and survives the
   correction in *direction*, but EVAL v1 was too small to establish it.
2. **P − P_shuffled collapses** to +0.0186 and is negative in the control
   scene. On the spent data there is no scene-level evidence that the geometry
   adds anything over a shuffled payload.
3. **The gain is not where it should be.** By family, near-tie ambiguous:

   | family | n | N | P | P_shuffled |
   |---|---|---|---|---|
   | RESOLVABLE | 311 | 0.318 | 0.328 | 0.322 |
   | NEGATIVE-A | 290 | 0.238 | 0.328 | 0.279 |
   | NEGATIVE-B | 290 | 0.238 | 0.328 | 0.283 |
   | CONTROL | 40 | 0.575 | 0.625 | 0.650 |

   On **RESOLVABLE** — the only family where a unique correct answer exists —
   placement moves accuracy by **+0.010**. Nearly the whole P − N gain comes
   from NEGATIVE-A/B (two *identical* objects, where the "correct" answer is
   object identity that the payload provably cannot see) and from CONTROLS.
   That is the signature of a score that is reading degree and coverage rather
   than geometry.
4. NEGATIVE-A and NEGATIVE-B are numerically identical, confirming mission 2's
   finding that no edge joins the two objects at the frozen radius.

## Restated position

- **Reach is solved**: ~65 % of ambiguous object nodes carry a checkable
  placement edge, pair survival 0.67, landmark share ~1.0, frame provider
  validated.
- **Significance on EVAL v1 is not established at the scene level**, and the
  headline P − N result is withdrawn as a significance claim.
- **The disagreement between P and P_shuffled is the real open question**, and
  the family table says the gain is concentrated where geometry cannot be the
  cause.
- Mission 3 therefore goes ahead exactly as the brief says: make the geometry
  carry the gain, with `P_presence` as the arm that isolates what the residual
  adds, on a DEV design sized in SCENES with a scene-level test that can
  actually resolve 0.03.

**Nothing in Stage 0 was used to choose anything.** Four scenes cannot support
a normal approximation; the pre-registration for every later stage fixes the
minimum scene count and the correction up front.
