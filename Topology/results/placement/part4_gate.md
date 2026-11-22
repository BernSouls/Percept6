# Task 4 MISSION — Part 4: the gate (measured, and it is UNEVALUATED)

## The rule

`sim' = sim · (1 + w · (support − 0.5))`, **ONE** pre-stated `w`
(`PLACEMENT_WEIGHT = 0.25`, in `parameters`). Neutral at support 0.5; can never
more than 1.5× or 0.5× a similarity; **a node with no checkable edge keeps its
similarity unchanged** (support `None` ⇒ no credit, no penalty).

## First: is there a usable truth-free gradient?

The mission requires `P(appearance-ambiguous | margin bin)` to be measured
**before** the gate is used, and says: if there is no usable gradient, report
P UNGATED and call the gate UNEVALUATED.

### What "appearance-ambiguous" can mean, measured

| observable | measured |
|---|---|
| exact-signature multiplicity | **536 distinct signatures on 536 nodes** — every signature is unique, ref side and query side alike. Exact-signature ambiguity is identically zero and can label nothing. |
| top node similarity ≥ 0.99 | **48.6 %** of nodes |
| exact similarity tie (margin = 0) | **4.5 %** of nodes |

So a useful ambiguity label has to be the *near-tie* one: top match scores
≥ 0.99 **and** the margin to the runner-up ≤ 0.05. Both are observable from the
graphs alone. **Correctness is not used anywhere** — it is unobservable at match
time, and no claim here is about "correct" matches.

### P(ambiguous | margin bin) — the gradient is NOT usable

5 pre-registered quantile bins, n = 11 567 node-view rows:

| margin bin | margin range | n | **P(ambiguous)** |
|---|---|---|---|
| 0 | 0.0000 – 0.0132 | 3128 | **0.3987** |
| 1 | 0.0133 – 0.0309 | 1493 | **0.3108** |
| 2 | 0.0309 – 0.0399 | 2317 | **0.8166** |
| 3 | 0.0399 – 0.0834 | 2304 | **0.2782** |
| 4 | 0.0834 – 1.0000 | 2311 | **0.0000** |

P(ambiguous) **UNGATED = 0.3674**.

**This is not monotone.** It falls from bin 0 to bin 1, then *jumps* to 0.8166 in
bin 2 before collapsing to 0.2782 and 0.0000. The reason is legible: a *large*
margin can only happen when the top similarity is high and the runner-up is far
below it, so bin 4's constraint `margin > 0.0834` forces `P(top ≥ 0.99)` down —
but bin 2 sits at the margin range where the similarities are tightly clustered
just under 1.0, which is exactly where near-ties live. The middle bin carries
most of the apparent ambiguity, and the extremes carry little.

A gate keyed on the margin would therefore **suppress placement support exactly
where ambiguity is least (bin 4) and permit it in the middle bin where ambiguity
is worst** — the opposite of the intent.

## Verdict

**The gate is UNEVALUATED.** There is no usable truth-free gradient:
`monotone_falling = False`, and although the spread is large (0.8166) it is
non-monotone, so no threshold on the margin maps to "ambiguous".

Per the mission's own instruction, the reported quantity is
**P(ambiguous) UNGATED = 0.3674**, and `PLACEMENT_GATE_ENABLED` **stays False**.
The modifier is therefore evaluated **ungated** in Part 6.

This is recorded as a finding, not worked around: the margin is not a proxy for
appearance ambiguity in these graphs, and tuning the bin count or the ambiguity
threshold until it looks monotone would be exactly the sweep the mission
forbids.
