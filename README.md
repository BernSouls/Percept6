# Percept6

A perception stack that turns one image into a graph of placed, keyed parts, and recognises objects
by matching those graphs.  The numerical stack is pinned in `requirements.txt` (numpy 2.4.6,
scipy 1.17.1, pillow 13.0.0); every read records the CPU and the versions in its manifest.

## The shipped retriever

`matching/placement_recognise.py` is the product.  `recognise(query, gallery)` ranks a gallery's
objects for one query image:

```python
from matching.placement_gallery import build_gallery
from matching.placement_recognise import recognise

gallery = build_gallery(items, code_digest=..., content="colour_rings")
result = recognise("query.png", gallery)                  # the full path: door, then the matcher
door_only = recognise("query.png", gallery, matcher=False)  # the door's own order, nothing scored
result.ranking, result.shortlist, result.scores
```

The **fused door** (`door.kind = "fused3"`) is a reciprocal-rank fusion of three channels — the
counted place matches, the whole-image appearance descriptor, and the six CIELAB ring means each
place carries (`content.kind = "colour_rings"`, `matching/placement_content.py`) — with the colour
channel at weight 0.5 and `k = 60`.  It hands the top **8** objects to the M19 matcher
(`matching/joint_mcs.py`).  Everything above is frozen in `matching/placement_bundle.json` and
refused at load time if this tree cannot honour it; nothing falls back silently.

Measured (`MISSION30_REPORT.md`, A1; 80 objects, chance 0.100): the fused door reads **0.675 /
0.650** recall@8 on the FIT split at the two read views, against **0.550 / 0.550** for the
two-channel door it replaces, with no query losing a hit.  The all-80 read is **23/80 = 0.2875
[0.188, 0.388]** (`MISSION31_REPORT.md` §2b; M27 shipped 19/80 = 0.2375, MDE 0.121, so it is an
estimate).  Chance is 0.0125 top-1 and 0.100 recall@8.

**M32 measured both channels and shipped neither** (`MISSION32_REPORT.md`).  Colour and shape were
both read through a transform that removed their variation: the colour term is a cosine on a
unit-normalised ring vector dominated by lightness (100 % of pairs ≥ 0.99, truth−impostor gap
0.000026) and `var_x` is pinned at `_parabola`'s clip (one distinct value over 46,059 places).
Replacing the cosine with a real ring distance (`matching/placement_content.d2`, `λ_L = 0.25`,
`σ` = 0.3505 from a seeded label-free pool sample) removed the flattening — 41.6 % ≥ 0.99, gap
0.01201 — but cost rank, because the shipped "colour" channel was ranking on lightness; and the
unclipped curvature (`sharp_*`, `curv_*`, now graded columns on the record, keys byte-identical)
lifted the same-class shape gauge from 0.505 to 0.596, short of the +0.10 bar.  The channel blend
(`matching/placement_blend.py`) therefore ships at **w_shape = w_colour = 0**, which reproduces the
deployed score exactly, and the bundle records why (`channels` block).

## Layout

* `matching/` — the shipped mechanisms: door, places, matcher, bundle, records.
* `core/`, `lenses/` — registries, constitutions, detectors.
* `Topology/` — thin harnesses that import the core; they measure, they do not define.
* `tests/` — every shipped behaviour is pinned by a test that goes red when it is reverted.
