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

The **bag door** (`door.kind = "bag"`, M35) scores an object by the HISTOGRAM of its places'
measurements — an object is a distribution, not a max.  Three blocks read the record alone: the ab
plane of ring 1 (5×5), the ab plane of ring 2 (5×5) and ring-1 lightness (6 bins); each block is
L1-normalised then square-rooted, the group score is the mean of the blocks' dot products, and the
object score is the max over the gallery's stored views (`matching/placement_bag.py`, bin edges in
`matching/placement_bag_edges.json` with provenance).  It hands the top **8** objects to the M19
matcher (`matching/joint_mcs.py`), which ships as flag `verify` because it could not be shown to add
its +0.075 on this door.  `matching/placement_recognise.recognise_class(query, gallery)` gives the
class answer (mean of a class's top-2 instance scores, own object excluded).  The fused three-channel
door (`door.kind = "fused3"`, RRF of counted + appearance + colour at 0.5, `k = 60`) stays
implemented as the legacy path.  Everything above is frozen in `matching/placement_bundle.json` and
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

**M33 registered the curvature channel at OBJECT level** (`Topology/m33_curvature.py`; four columns,
kernel form, fourth door channel at half vote) and measured it on the same cells: **+0.0375 mean FIT
recall@8** (3 objects up, 0 down, at one view; 0/0 at the other), object-clustered CI **[0.000,
0.0875] including zero**, against a +0.050 bar — so it does not ship either, and the mechanism is
the COLUMNS rather than the aggregation: **80–99 % of every curvature column's variance is within an
object** (ICC 0.009–0.197), so the profile transfers across views (decile correlation 0.56–0.79) but
the identity does not.  The record file is now **committed at 38.03 MB** (nine graded columns at
float32, everything the door buckets on exact), which turns a fresh checkout's eight-minute rebuild
into **21 s** of loading with 80/80 shortlists identical at both read views and a cosine delta of
0.0 (`MISSION33_REPORT.md`).

**M34 composed places into FRAGMENTS and shipped the channel** (`MISSION34_REPORT.md`).  The
mission's grouping rule does not exist: on a real graph the front end emits two disjoint node
universes and **0 of 2,686 edges touch a place node** — all 474 places are isolated — so the four
edge classes it named group places zero times.  The substitution is registered in writing: a place
joins the fragment of its nearest `ContourFragment` anchor, which *wires the region machinery to the
places* for the first time.  The descriptor (`matching/placement_fragment.py`) is twenty-two
frame-relative numbers — size, aspect, member arrangement measured from the fragment's own principal
axis, pooled curvature and the six CIELAB rings — in a class registered in ranges, weights and
population axes, and it answers **0.0** when that class is absent.  Measured: **0.725 / 0.637**
channel-alone recall@8 on the two FIT views (mean 0.681, bar 0.675, chance 0.100) and **18/31** on
M31's same-class losses (bar 15/31, colour 22/30).  Fused into the door as a **fifth channel at the
stated discount 0.5** it reads **+0.125 at `-035-045` (5 objects up, 0 down), 0.000 at `-045-180`**,
pooled **+0.0625 with an object-clustered CI [0.0125, 0.1125]**, and re-ranking inside the cached
eight it fixes **5, breaks 0** (all five were same-class impostors).  The anchors a fragment needs
are not on the place record, so they ship as data (`data/placement_anchors_m34.npz`, 588 KB) and the
door resolves them from the image path — an image outside the file is computed, never zeroed.
`Topology/m34_wire.py` proves the shipped door returns the measured order (6/6, with the anchors resolved from the image path), and the shape
channel as a re-ranker is a **named ceiling**: FIT top-1 **−0.125, CI [−0.225, −0.025]** against a
+0.075 bar, because its order inside the eight is uncorrelated with the matcher's (Spearman 0.016).

`Topology/m34_audit.py` also answers, with counts, whether the tree's machinery is wired: **83.1 % of
78,587 lines are reachable** from a shipped entry point, **13.1 % (19 modules, 10,291 lines) is
orphan** — including 3,185 lines of fragment machinery nobody had wired — and of the 24 edge types
the front end emits, **16 reach no module that turns a graph into a number** while **474 places reach
no edge at all**.  That verdict is in `MISSION34_REPORT.md` §7, and it is the reason the fragment
channel ships wired rather than measured and shelved.

**M35 made the object the unit and re-chose the door by measurement** (`MISSION35_REPORT.md`).  A
colour histogram over an object's places recalls **1.000 / 0.925** of the FIT half at `-035-045` /
`-045-180` (chance 0.100) against **0.800 / 0.650** for the fused door that shipped in M34 — a paired
**+0.2375, CI [0.1375, 0.3375]**, confirmation +0.300, on a door of ONE channel.  Adding the old
channels back *hurts* (0.950 / 0.800): RRF can only dilute a channel that is already right.  The
four-block descriptor (COLOUR / SHAPE / LAYOUT / CFG, 11 blocks, 745 numbers) was registered before
any number and tested for CLASS: geometry alone — place layout in the object's own frame plus the
neighbourhood shell profile, **no appearance term** — reads class AUC **0.679 [0.641, 0.716]** and
**0.622 [0.583, 0.659]** at the two hard views, so **geometry does cluster by class**, and it beats
colour exactly where colour fails (**car** 0.90 vs 0.51; **cup** 0.80 vs 0.69) while losing where
colour is strong (**pear** 0.54 vs 0.90).  The blocks are complementary (ALL − COLOUR **+0.053** and
**+0.055**, both CIs excluding zero) and ALL is the best class arm at every view
(0.806 / 0.791 / 0.783).  The class output reads **0.650 / 0.575 / 0.500** with CIs that include
M23's 0.500 — shipped, not claimed as an improvement.  The fragment channel was re-measured **on top
of the new door** and adds **+0.000** FIT recall@8 (bar +0.025), so its weight is **0.0** (code and
anchors stay); its own ablation withdraws M34's "composition" reading — the eight arrangement fields
alone are at chance (0.275) and the six **part-level colour means** alone beat all 22 (0.825 vs 0.700
at `-035-045`).  The matcher's contribution on this door is a **named ceiling**: 640 uncached pairs
≈ 6 h of one core, not finished in the window, so the door's order is final and the matcher is
`verify`.  The tree is closed too: **12 orphan modules, 6,143 lines** moved to `archive/` (suite green
after each of three batches) and **6 left in place** because moving them would break a kept import
(`lenses/` is excluded).  Limits: 80 objects, plain backgrounds, no real scenes; a lighting change
moves every colour bin; the next lever is a second dataset whose classes do not differ in colour.

## Layout

* `matching/` — the shipped mechanisms: door, places, matcher, bundle, records.
* `core/`, `lenses/` — registries, constitutions, detectors.
* `Topology/` — thin harnesses that import the core; they measure, they do not define.
* `tests/` — every shipped behaviour is pinned by a test that goes red when it is reverted.
