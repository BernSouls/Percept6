# Part 6 — PRE-REGISTRATION (written before the run)

## Scene count (the power floor)

Model graphs: 24 scene-graphs = 4 families × 6 view steps, of which 5 non-base
views are scored, so **20 scene-graphs** enter the arms table.

| family | scenes in the model | why |
|---|---|---|
| RESOLVABLE | 1 | the positive test |
| NEGATIVE-A | 2 | two identical objects, no occlusion |
| NEGATIVE-B | 1 | same scene, cross-object edges |
| CONTROL | 2 | ordinary objects |

`SCENES_PER_FAMILY = {"RESOLVABLE": 1, "NEGATIVE-A": 2, "NEGATIVE-B": 1, "CONTROL": 2}`

## MDE, stated before the run

Clusters are **scene-graphs** (`family|scene|view`), so n_eff is 20 and the
bootstrap resamples them. With a binary accuracy read-out the worst-case
within-cluster variance is 0.25, giving a conservative standard error

    SE_max = sqrt(0.25 / n_nodes) x sqrt(1 + (n_eff - 1) x rho)

and the pre-registered MDE is `2.8 x SE`. The EXPECTED effect is the difference
placement makes on ambiguous nodes; Part 3 measured the payload's separation at
AUC 0.81-0.92 (d' 1.21-1.97), so a plausible accuracy gain is **+0.05 to +0.15**.
If the realised MDE exceeds 0.05, the ambiguous-node contrast is expected to be
INCONCLUSIVE and will be reported as such, not as "no effect".

## Pre-stated decisions (no tuning after the fact)

- `PLACEMENT_WEIGHT = 0.25`, one value, no sweep.
- Modifier is **ungated** (Part 4 measured no usable truth-free gradient).
- P-shuffled: same links, geometry permuted among edges of the SAME type,
  deterministic seed 7, derangement required (no edge keeps its own geometry).
- W: endpoints swapped, degree preserved, canonical sort BEFORE the seeded
  shuffle (frozenset order is process-salted).
- K is NOT run: 0a found the walker's admission test is not mapping-dependent,
  so it is not a placement candidate.
- Ambiguity label: observable near-tie rule from Part 4, fixed before the run.
- Verdict rule: INCONCLUSIVE unless the CI excludes effects above the MDE.
