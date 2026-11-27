# Mission 26 -- registration, written before any door number exists

State at the time of writing: the place records are still being rebuilt in this sandbox
(``Topology/m26_records.py --build``); **no variant of any door has been scored on any cell**, and
no selection or confirmation number exists.  What follows is fixed now.

## 1. What the door is being asked

The shipped door compares two places by EXACT equality of ``(lens, quantised tuple)`` -- one dict
probe, ``index.by_key.get(key)`` -- and then ranks the bucket by the frozen configuration
distance.  The matcher already compares places by graded similarity.  Item 1's census counts how
much of the place population that equality throws away; items 2-3 replace it with the registry's
own graded kernel, vectorised, and the question is whether the shortlist ceiling moves.

## 2. The graded door (item 3), fixed here

For one query place ``q`` and one pool place ``p`` of the same lens:

```
    s(q, p) = kernel(class_=lens, q.quantised, p.quantised)  x  exp(-d2(q, p) / 2)
```

* ``kernel`` is ``matching/placement_kernel.py``: the registry's ``CONSTITUTIONAL_RANGES`` with
  ``sigma_i = ranges[i] / 3``, the registered slot weights, the circular theta cycle (12 bins),
  the absent-slot renormalisation, and ``core.slots.class_slot_distance`` overrides -- the same
  arithmetic as ``core.types._similarity_gaussian``'s quantised branch, asserted to 1e-9.
* ``d2`` is the Mission-21 frozen-ruler configuration distance already in the door
  (``placement_frontdoor.cfg_distance``: mean squared standardised distance over the overlap mask).
* **The config scale is fixed at ``exp(-d2 / 2)``** -- one half, no fitted value -- so a place
  whose configuration differs by one standard deviation on every slot loses ~0.61.

Per query place, the BEST place PER OBJECT (not a pooled top-n): ``s(q, o) = max_p in o s(q, p)``.

Distinctiveness, label-free: an object counts as covered by ``q`` when ``s(q, o) >= tau``, with
**``tau = 0.5`` fixed here**; ``df(q)`` is the number of the gallery's ``N`` objects covered, and

```
    w(q) = log((N + 1) / (df(q) + 1))
```

The object's score is weighted query coverage -- the weight of the query places it covers over the
total weight of the query's places that can vote at all (live configuration, ``>= 2`` slots):

```
    score(o) = sum_{q : s(q,o) >= tau} w(q)  /  sum_q w(q)
```

Two constants appear in the whole door: ``tau = 0.5`` and the ``1/2`` in the config exponent.  Both
are fixed in this paragraph, before a single query was scored, and neither is fitted on any cell.

## 3. The variants (item 4), each defined once

| id | channel | definition |
|---|---|---|
| a | counted door (shipped) | ``count_scores(index, query, n_hits = 8)`` at the reference view: the exact-key probe, unchanged |
| b | appearance (shipped) | the fixed 4x4x4 + 12-bin descriptor, cosine |
| c | RRF(a, b) | the shipped door, k = 60 |
| d | graded signature only | section 2 with ``s = kernel`` (no configuration factor) |
| e | configuration only | section 2 with ``s = exp(-d2/2)`` (no signature factor) |
| f | the graded door | section 2 exactly |
| g | f without distinctiveness | section 2 with ``w(q) = 1`` |
| h | f with pooled top-8 | section 2, but each query place covers the objects of its 8 best pool places over ALL objects (the counted door's pooling), then weighted coverage |
| i | graph-native global layout | from the record only: lens shares (3), log2 place-scale histogram (6 fixed bins over [-1, 3] octaves), extent-normalised radial histogram (6 bins of r / r_max) and axial-orientation histogram (6 bins of theta over [0, pi)); L2-normalised, cosine |
| j | RRF pairs | ``RRF(b, f)``, ``RRF(b, f, i)``, ``RRF(b, a, f)`` -- summed ``1/(60 + rank)`` over the lists, no weight |

Union oracle: the truth is in the top-8 of ANY of a..i.  Thin objects: the truth's place count
below 50 % of the median place count of the gallery.

## 4. Cells

* **selection**: the 40 FIT objects, at query views ``-035-045`` and ``-045-180`` (80 query cells);
  ``-022-000`` is also scored and reported as the saturation check, not as a selection cell.
* **confirmation**: the 40 DEV + FRESH objects, at the same two views (80 query cells).
* gallery: every object at its reference view ``-000-000`` (the shipped k = 1 bundle), 80 objects.
* chance: recall@8 = 8/80 = **0.100**, top-1 = 1/80 = **0.0125**.

## 5. The selection rule (item 5)

Choose on the FIT-object cells only; ties go to the variant with FEWER channels.  A variant ships
if BOTH hold:

1. its FIT-object recall@8 beats the shipped door (c) by **>= 0.075**; and
2. its confirmation-object paired difference against (c) is **>= -0.025**, object-clustered (10,000
   resamples, seed 20261118), with the interval reported.

**MDE, stated before the run** (the Mission-23 simulation, ``archive/Topology/placement_m23_power.py``,
two queries per object):

| objects | per object | MDE w = 0 | MDE w = 0.5 | MDE w = 1.0 |
|---|---|---|---|---|
| 40 (confirmation) | 2 | 0.199 | 0.183 | 0.170 |
| 80 (both splits together) | 2 | 0.134 | 0.132 | 0.108 |

The confirmation test is therefore a **non-inferiority guardrail, not a proof**: at 40 objects the
interval cannot resolve ±0.025, so a pass means "no evidence of a drop", not "no drop".  This is
written down before the numbers so that a pass cannot be reported as more than it is.

## 6. The outcome ladder (item 6)

A. a graded variant passes item 5 -> it becomes the bundle's door (``door.kind = "graded"``), with
   the counted door kept and pinned (M24/M25 equivalence: door 80/80, matcher 44/44).
B. nothing passes, but the union oracle >= 0.80 on the selection cells -> the ceiling is fusion:
   ship the best fixed RRF pair that passes item 5, else NAME the pair to fuse and say why it does
   not pass.
C. nothing passes, union < 0.80, and item 1 shows >= 25 % of query places dropped -> price the
   record migration (sum of the records' ``sec``), run a SUBSET test first (8 objects, both views,
   population payload + dropped places' signatures in the record, door by population overlap); run
   the full migration only if the subset gains >= 0.075 recall@8.
D. nothing helps -> ADAPTIVE S: S = 8 when the top-8 door margin is clear, S = 16 otherwise, with
   the margin rule fixed on the FIT-object cells and reported with the ceiling it implies.

## 7. The matcher step (item 7)

Only after the door is fixed: list the (query, candidate) pairs for the 80 x ``-035-045`` cell that
are NOT in the pair cache, state the count and the estimated minutes, run exactly those with a
budget, and report assembled top-1 with a paired interval against 0.250 as an estimate.  Queries
that still lack a score are listed, never zeroed.
