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
two-channel door it replaces, with no query losing a hit.

## Layout

* `matching/` — the shipped mechanisms: door, places, matcher, bundle, records.
* `core/`, `lenses/` — registries, constitutions, detectors.
* `Topology/` — thin harnesses that import the core; they measure, they do not define.
* `tests/` — every shipped behaviour is pinned by a test that goes red when it is reverted.
