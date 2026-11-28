"""[MISSION 36 / PART B] THE DENSE LOCAL-DIFFERENCE FIELD.

WHAT THIS IS.  A node wherever the LOCAL CONTENT DIFFERS from its neighbourhood,
at several scales, at every location the difference clears a threshold -- not on a
lattice, and with no segmenter anywhere.  A uniform region produces nothing; a
shading gradient, a colour edge, a texture boundary and a contour all produce
nodes.

THE OPERATION, and why each part of it is the right one:

  * CENTRE-SURROUND.  The difference is measured as a difference of Gaussians of
    the local energy, exactly the retinal operation (`perception.filters.
    center_surround`): a zero-mean kernel, so a uniform field gives exactly 0
    (Kuffler 1953).  Dividing by the local mean (Weber) is what keeps the
    response to the SAME surface stable under a change of illumination (Turner et
    al. 2018): the node exists for a ratio, not for a brightness.
  * THE RADII ARE THE FILTER'S OWN.  Every radius below is a multiple of the
    Gabor filter's sigma at that scale (`GABOR_SIGMA_X_FRAC / sf`), never a fixed
    pixel count, so the same physical structure is measured the same way at any
    image size.
  * QUADRATURE ENERGY, POOLED (B2).  The measurement at a scale is the quadrature
    pair's squared sum over the bank and a Gaussian pool (`POOL_FRAC * sigma`),
    so a response survives a small shift without an interest point: a complex
    cell answers its orientation ANYWHERE in a small region (Adelson & Bergen
    1985).
  * OVERCOMPLETE.  Three scales, all orientations, both colour opponents; a node
    is minted per local maximum per scale, so supports overlap and the field
    covers the surface rather than sampling points on it (Olshausen & Field 1997;
    Nauhaus 2016).
  * ACTIVITY IS SPARSE, COVERAGE IS NOT.  Only positions where the difference
    clears the threshold are nodes (Tang et al. 2018: few units active), and the
    count is a property of the IMAGE, not of a grid -- `nodes_per_image` on a
    uniform patch is 0 and on a textured one is large (see `selftest_contrast`).

WHAT IT IS NOT.  No region growing, no patch boundary decision, no merging, no
clustering: a threshold on a local difference measurement, then a local maximum.
Nothing here decides what an object is.

WHAT IT CARRIES.  One `DENSE_FIELD` signature per node, holding the local content:
the dominant orientation, the orientation coherence, the Weber contrast that
minted it, the local opponent colour and luminance, and the scale.  The class is
registered in `core.registries` (`POPULATION_SLOTS`, `POPULATION_AXES`,
`CONSTITUTIONAL_RANGES`, `CIRCULAR_CYCLE_LENGTHS`), so the tree's own kernel
compares it (`core.types._similarity_gaussian`); an unregistered class scores 0.0.

POSITION IS NOT CLAIMED TO BE STABLE.  A dense node is carried by the place it
attaches to (`matching/placement_join.py`), which is why nothing downstream treats
its coordinates as an identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The class name every dense node's signature carries.
DENSE_CLASS = "DENSE_FIELD"

#: Scales used, as INDICES into the Gabor bank's sf axis.  The bank's sf band is
#: geometric, so these are roughly one octave apart; the sigma of each is
#: `GABOR_SIGMA_X_FRAC / sf` (9.3 / 3.9 / 1.6 px at the 256-px reference).
DENSE_SCALES: Tuple[int, ...] = (0, 2, 4)

#: The pool radius, as a fraction of the scale's own sigma (B2).
POOL_FRAC: float = 0.5
#: Centre and surround radii, as fractions of the scale's own sigma (B1).
CENTRE_FRAC: float = 0.5
SURROUND_FRAC: float = 1.5
#: A node is minted where the normalised difference reaches this.  A FIXED
#: constant, not a quantile: a per-image quantile would mint the same number of
#: nodes on every image, which is exactly the lattice defect B4 tests for.
DENSE_TAU: float = 0.25
#: The local maximum footprint, as a fraction of the scale's sigma.  Two nodes of
#: one scale are at least this far apart; nodes of DIFFERENT scales may coincide
#: (their supports differ, which is the overcomplete part).
PEAK_FRAC: float = 0.75

#: [M36/B1] THE SAMPLING RULE THAT MAKES COVERAGE DENSE.  A local maximum alone is
#: a SPARSITY device: on 256x256 it produced ~514 nodes against the cell's 518
#: places -- one dense node per place, so the second level added nothing to the
#: first.  The mission's distinction is sparse CODING vs sparse SAMPLING: the
#: sampling must be dense.  So the node set is the greedy packing of the
#: ABOVE-THRESHOLD sites, taking the strongest first and rejecting any site closer
#: than MIN_SPACING_FRAC * sigma to a site already taken at that scale.  That is
#: still one threshold on a local difference measurement plus a spacing rule --
#: no region, no merge, no boundary decision -- and it guarantees that no content
#: region wider than the scale is left without a node.
#: PRE-REGISTERED at 1.5 (a node every one-and-a-half sigmas), not tuned.
MIN_SPACING_FRAC: float = 1.5

#: The pre-fix control: peaks only, no packing (kept so the report can name both).
DENSE_PEAK_ONLY: bool = False
#: Floor on the Weber denominator, in the same units as the pooled energy.
_WEBER_EPS: float = 1e-3
#: [M36/B1 FIX] THE DENOMINATOR'S FLOOR, as a fraction of the frame's MEDIAN
#: surround energy at that scale.  Without it the Weber ratio divides by a
#: surround that can be ~0 in a smooth area, and sensor noise is amplified into
#: "content": measured on cup1-035-045, the SMOOTHEST decile of 32x32 patches
#: inside the object yielded 25.6 nodes per patch while the MOST TEXTURED decile
#: yielded 8.3 -- the field read the absence of content as content.  With the
#: floor the ratio cannot exceed the frame's own typical energy scale, so a
#: smooth area stays quiet and a textured one does not.
#: PRE-REGISTERED at 0.5 (half the frame's median energy) -- not tuned.
DENOM_FLOOR_FRAC: float = 0.5
#: HOW THE DIFFERENCE IS NORMALISED (see `normaliser_choice`):
#:   "local"  -- divide by the local surround mean (Weber).  Invariant to a local
#:               illumination change; normalises away an image's global contrast,
#:               which is why a globally busier frame can give FEWER peaks.
#:   "global" -- divide by the frame's median surround (one number per scale).
#:               Invariant to an overall gain change and keeps the image-to-image
#:               contrast ordering.
#:   "none"   -- no normalisation (the control: a gain change moves the count).
#: The default is set by `DENSE_NORMALISER` in parameters; `normaliser_choice`
#: documents the measurement that chose it.
DENSE_NORMALISER: str = "local"


@dataclass(frozen=True)
class DenseNode:
    """One dense node: where it is, at what scale, and what it holds."""

    x: float
    y: float
    scale_index: int
    sigma_px: float
    support_px: float
    #: the local difference that minted it, in [0, inf) -- Weber-normalised
    activity: float
    #: the registered slot values, IN SLOT ORDER (core.slots.SLOT_NAMES[DENSE_CLASS])
    slots: Tuple[float, ...]
    #: the same measurements BEFORE quantisation (the population's centres)
    values: Tuple[float, ...]

    def anchor(self) -> Tuple[float, float, float]:
        """The node's anchor tuple: position plus the scale it was measured at."""
        return (float(self.x), float(self.y), float(self.scale_index))

    # --- the named slots, read in ONE place rather than by index at the call sites ---
    @property
    def theta(self) -> float:
        """Dominant local orientation, AXIAL, in [0, pi)."""
        return float(self.values[0])

    @property
    def coh(self) -> float:
        """Orientation coherence: the vector strength of the node's orientation marginal."""
        return float(self.values[1])

    @property
    def energy(self) -> float:
        """The Weber-normalised local difference that minted the node."""
        return float(self.values[2])

    def slot(self, name: str) -> float:
        """One slot by NAME (the layout's order is the registry's business, not a caller's)."""
        from core.registries import POPULATION_SLOTS
        names = [s.name for s in POPULATION_SLOTS[DENSE_CLASS]]
        return float(self.values[names.index(str(name))])


@dataclass
class DenseField:
    """The field of one image: its nodes plus the maps they were read from."""

    nodes: Tuple[DenseNode, ...]
    #: per-scale (activity, energy, coherence) maps, for the measurements in
    #: `coverage`/`stability` and for a test that wants to look at the field.
    maps: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
def _planes(rgb: Any) -> Dict[str, np.ndarray]:
    """The front end's own planes: lum, rg, by (one luminance path, as the chains use)."""
    from perception.chain_flanks import planes_from_rgb
    pl = planes_from_rgb(np.asarray(rgb))
    if pl is not None and all(k in pl for k in ("lum", "rg", "by")):
        return {k: np.asarray(pl[k], dtype=np.float32) for k in ("lum", "rg", "by")}
    from perception.filters import opponent_channels
    opp = opponent_channels(np.asarray(rgb))
    return {"lum": np.asarray(opp[:, :, 0], dtype=np.float32),
            "rg": np.asarray(opp[:, :, 1], dtype=np.float32),
            "by": np.asarray(opp[:, :, 2], dtype=np.float32)}


def _front_end_scale(h: int, w: int) -> float:
    """The bank's scale for a frame: 1.0 at the 256-px reference (the tree's convention)."""
    return max(float(min(h, w)) / 256.0, 1e-6)


def _normaliser() -> str:
    """The normaliser in force: the parameter if declared, else the module default."""
    try:
        import parameters as P
        v = str(getattr(P, "DENSE_NORMALISER", "") or DENSE_NORMALISER)
    except Exception:  # pragma: no cover
        v = DENSE_NORMALISER
    return v if v in ("local", "local_none", "global", "none") else DENSE_NORMALISER


def _norm01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _local_maxima(activity: np.ndarray, sigma: float) -> np.ndarray:
    """A boolean mask of local maxima, one footprint per scale (content-driven spacing)."""
    from scipy.ndimage import maximum_filter
    size = int(max(3, 2 * int(math.ceil(PEAK_FRAC * float(sigma))) + 1))
    return (activity >= maximum_filter(activity, size=size, mode="nearest")) & \
           (activity >= DENSE_TAU)


def _pack(activity: np.ndarray, sigma: float, tau: float,
          spacing_frac: Optional[float] = None, peak_only: Optional[bool] = None) -> np.ndarray:
    """The node mask at one scale: greedy packing of the above-threshold sites.

    Order is by activity, descending, so a local maximum always wins its own
    neighbourhood; a site is rejected when a site already taken is closer than
    ``spacing_frac * sigma`` in EUCLIDEAN distance.  Deterministic: ties in
    activity are broken by (y, x), and the grid test is a fixed integer radius.
    """
    sf = MIN_SPACING_FRAC if spacing_frac is None else float(spacing_frac)
    if bool(DENSE_PEAK_ONLY if peak_only is None else peak_only) or sf <= 0.0:
        return _local_maxima(activity, sigma)
    cand = np.asarray(activity, dtype=np.float64) >= float(tau)
    if not cand.any():
        return np.zeros(activity.shape, dtype=bool)
    ys, xs = np.nonzero(cand)
    vals = activity[ys, xs].astype(np.float64)
    order = np.lexsort((xs, ys, -vals))
    r = float(sf) * float(sigma)
    r2 = r * r
    taken = np.zeros(activity.shape, dtype=bool)
    rad = int(math.ceil(r))
    if rad <= 0:
        return cand
    h, w = activity.shape
    for i in order.tolist():
        y = int(ys[i]); x = int(xs[i])
        if taken[y, x]:
            continue
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        win = taken[y0:y1, x0:x1]
        if win.any():
            wy, wx = np.nonzero(win)
            dy = (wy + y0) - y
            dx = (wx + x0) - x
            if (dy * dy + dx * dx).min() <= r2:
                continue
        taken[y, x] = True
    return taken


def dense_field(rgb: Any, *, scales: Sequence[int] = DENSE_SCALES,
                tau: Optional[float] = None) -> DenseField:
    """The dense local-difference field of one image.

    Returns the nodes (canonically ordered by scale, then y, then x) and the maps
    they were read from.  Pure function of the pixels and the parameters.
    """
    from scipy.ndimage import gaussian_filter
    from perception.filters import GaborBank, compute_responses

    img = np.asarray(rgb)
    if img.ndim == 3:
        h, w = int(img.shape[0]), int(img.shape[1])
    else:  # pragma: no cover - a single-channel frame still has a field
        h, w = int(img.shape[0]), int(img.shape[1])
    planes = _planes(img)
    lum = _norm_planes_lum(planes["lum"])
    scale_fe = _front_end_scale(h, w)
    bank = GaborBank.build_default(scale_fe)
    resp = compute_responses(bank, lum)
    energy_all = np.asarray(resp["complex"], dtype=np.float32)     # (nt, nsf, H, W)
    nt = int(energy_all.shape[0])

    thr = float(DENSE_TAU if tau is None else tau)
    nodes: List[DenseNode] = []
    maps: Dict[str, Any] = {}
    stats: Dict[str, Any] = {"scales": [], "tau": thr, "n_theta": nt}

    for isf in scales:
        isf = int(isf)
        if isf < 0 or isf >= energy_all.shape[1]:
            continue
        sigma = float(bank_sigma(bank, isf))
        energy = np.asarray(energy_all[:, isf], dtype=np.float32)   # (nt, H, W)
        summed = energy.sum(axis=0)
        pool = float(max(0.8, POOL_FRAC * sigma))
        pooled = gaussian_filter(summed, sigma=pool, mode="nearest")
        centre = gaussian_filter(pooled, sigma=max(0.6, CENTRE_FRAC * sigma), mode="nearest")
        surround = gaussian_filter(pooled, sigma=max(1.2, SURROUND_FRAC * sigma), mode="nearest")
        _med = float(np.median(surround))
        if _normaliser() == "global":
            web_den = np.full_like(surround, max(_med, _WEBER_EPS))
        elif _normaliser() == "none":
            web_den = np.ones_like(surround)
        elif _normaliser() == "local_none":
            # the pre-fix control: no floor at all
            web_den = np.maximum(surround, _WEBER_EPS)
        else:
            web_den = np.maximum(surround, max(_WEBER_EPS, DENOM_FLOOR_FRAC * _med))
        act_lum = np.abs(centre - surround) / web_den

        # colour: the same centre-surround on each opponent plane, normalised by
        # its own local magnitude (a ratio, so illumination drops out).
        act_col = np.zeros_like(act_lum)
        for key in ("rg", "by"):
            ch = np.abs(np.asarray(planes[key], dtype=np.float32))
            pch = gaussian_filter(ch, sigma=pool, mode="nearest")
            cch = gaussian_filter(pch, sigma=max(0.6, CENTRE_FRAC * sigma), mode="nearest")
            sch = gaussian_filter(pch, sigma=max(1.2, SURROUND_FRAC * sigma), mode="nearest")
            _gc = float(np.median(sch))
            if _normaliser() == "global":
                cden = np.full_like(sch, max(_gc, _WEBER_EPS))
            elif _normaliser() == "none":
                cden = np.ones_like(sch)
            elif _normaliser() == "local_none":
                cden = np.maximum(sch, _WEBER_EPS)
            else:
                cden = np.maximum(sch, max(_WEBER_EPS, DENOM_FLOOR_FRAC * _gc))
            act_col = np.maximum(act_col, np.abs(cch - sch) / cden)

        activity = np.maximum(act_lum, act_col).astype(np.float32)
        mask = _pack(activity, sigma, float(thr))

        # local colour/luminance read through the same pool, so a node's content
        # is the surface it covers and not a single pixel.
        lum_s = gaussian_filter(lum, sigma=pool, mode="nearest")
        rg_s = gaussian_filter(np.asarray(planes["rg"], dtype=np.float32), sigma=pool, mode="nearest")
        by_s = gaussian_filter(np.asarray(planes["by"], dtype=np.float32), sigma=pool, mode="nearest")
        # the orientation marginal, pooled per channel so a single noisy cell
        # cannot decide the node's orientation
        theta_marg = np.stack([gaussian_filter(energy[it], sigma=pool, mode="nearest")
                               for it in range(nt)], axis=0)
        theta_bank = np.asarray(bank.theta, dtype=np.float64)

        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            p = np.maximum(theta_marg[:, y, x].astype(np.float64), 0.0)
            tot = float(p.sum())
            if tot <= 1e-12:
                coh = 0.0
                theta_c = 0.0
            else:
                p = p / tot
                c = float((p * np.cos(2.0 * theta_bank)).sum())
                s = float((p * np.sin(2.0 * theta_bank)).sum())
                coh = float(min(1.0, math.hypot(c, s)))
                theta_c = float(0.5 * math.atan2(s, c)) % math.pi
            vals = (
                float(theta_c),                                  # theta        (rad, axial)
                float(coh),                                      # coh          (0..1)
                _norm01(float(activity[y, x])),                  # energy       (Weber contrast)
                _norm01(float(lum_s[y, x])),                     # lum          (0..1)
                float(np.clip(rg_s[y, x], -1.0, 1.0)),           # rg           (-1..1)
                float(np.clip(by_s[y, x], -1.0, 1.0)),           # by           (-1..1)
                _norm01(float(np.log2(max(sigma, 1e-3)) / 5.0)),  # sigma        (log2 px over 2^5)
            )
            nodes.append(DenseNode(
                x=float(x), y=float(y), scale_index=isf, sigma_px=float(sigma),
                support_px=float(max(1.5, 1.5 * sigma)),
                activity=float(activity[y, x]),
                slots=_quantise_slots(vals), values=vals))
        maps[f"activity_{isf}"] = activity
        maps[f"mask_{isf}"] = mask
        stats["scales"].append({
            "scale_index": isf, "sigma_px": round(float(sigma), 3),
            "n_nodes": int(mask.sum()),
            "activity_mean": round(float(activity.mean()), 6),
            "activity_max": round(float(activity.max()), 6),
            "mask_frac": round(float(mask.mean()), 6),
        })

    nodes.sort(key=lambda n: (int(n.scale_index), float(n.y), float(n.x)))
    stats["n_nodes"] = len(nodes)
    return DenseField(nodes=tuple(nodes), maps=maps, stats=stats)


def _norm_planes_lum(lum: Any) -> np.ndarray:
    """Luminance onto [0, 1]: `planes_from_rgb` already adapts, but a caller's plane
    may be 0..255.  The bank is scale-free, yet the Weber denominator is not: one
    convention keeps `_WEBER_EPS` meaningful."""
    a = np.asarray(lum, dtype=np.float32)
    if float(a.max()) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def bank_sigma(bank: Any, isf: int) -> float:
    """The sigma, in px, of the bank's filters at one scale index."""
    import parameters as P
    sf = float(np.asarray(bank.sf, dtype=float)[int(isf)])
    return float(P.GABOR_SIGMA_X_FRAC / max(sf, 1e-9))


def _quantise_slots(values: Sequence[float]) -> Tuple[float, ...]:
    """The bins path: each value through its slot's own quantiser, by NAME."""
    from core import registries as R
    spec = R.POPULATION_SLOTS.get(DENSE_CLASS, ())
    if len(spec) != len(values):
        raise ValueError("DENSE_FIELD slot layout disagrees with the emitter "
                         "(%d slots registered, %d measured)" % (len(spec), len(values)))
    return tuple(float(R.quantise(s, float(v))) for s, v in zip(spec, values))


# ---------------------------------------------------------------------------
# MEASUREMENTS (B4, B5) -- in the module, not in a harness, so a test can hold them
# ---------------------------------------------------------------------------
def coverage(field_: DenseField, shape: Tuple[int, int],
             support_of: Optional[Any] = None) -> float:
    """The share of the frame inside SOME node's support.

    A support is a disc of radius `support_px` around the node; the union is
    computed on the pixel grid.  Reported over the whole frame, which is the
    conservative denominator (the object does not fill it).
    """
    h, w = int(shape[0]), int(shape[1])
    cover = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for n in field_.nodes:
        r = float(n.support_px if support_of is None else support_of(n))
        cover |= ((yy - n.y) ** 2 + (xx - n.x) ** 2) <= r * r
    return float(cover.mean())


def object_mask(rgb: Any, tol: float = 18.0) -> np.ndarray:
    """A BACKGROUND mask for the coverage measurement -- not a segmenter.

    ETH-80's objects sit on a near-uniform background, so "object" for a COVERAGE
    REPORT is "pixel differs from the frame's border colour by more than `tol`".
    Nothing mints a node from this mask: the field is a threshold on the local
    difference, independent of what an object is.  It exists because B4 asks for
    the share of OBJECT pixels covered, and that question needs a denominator.
    """
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    border = np.concatenate([img[0, :, :3], img[-1, :, :3],
                             img[:, 0, :3], img[:, -1, :3]], axis=0)
    bg = np.median(border, axis=0)
    d = np.abs(img[..., :3] - bg[None, None, :]).max(axis=-1)
    m = d > float(tol)
    if m.mean() < 0.02:                      # a frame whose border is not background
        m = d > float(np.percentile(d, 75.0))
    return m


def object_coverage(field_: DenseField, rgb: Any) -> float:
    """The share of OBJECT pixels inside some node's support (B4)."""
    img = np.asarray(rgb)
    h, w = int(img.shape[0]), int(img.shape[1])
    mask = object_mask(img)
    if not mask.any():
        return float("nan")
    cover = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for n in field_.nodes:
        cover |= ((yy - n.y) ** 2 + (xx - n.x) ** 2) <= float(n.support_px) ** 2
    return float((cover & mask).sum()) / float(mask.sum())


def content_variation(rgb: Any) -> float:
    """A measure of the IMAGE's own content variation, for B4's correlation.

    The mean absolute Laplacian of the luminance, in [0, 1]: a uniform frame is 0,
    a textured one is large.  Computed from the pixels, independent of the field,
    so it is a genuine external regressor rather than a restatement of the node
    count.
    """
    from scipy.ndimage import laplace
    lum = _norm_planes_lum(_planes(rgb)["lum"])
    return float(np.abs(laplace(lum, mode="nearest")).mean())


def content_matrix(field_: DenseField) -> np.ndarray:
    """The nodes' measured (pre-quantisation) content, one row per node."""
    if not field_.nodes:
        return np.zeros((0, 0), dtype=float)
    return np.asarray([n.values for n in field_.nodes], dtype=float)


def _axial_features(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The measured content of the matched rows, with the axial angle as its two components."""
    X = np.asarray(M[idx], dtype=float).copy()
    th = X[:, 0]
    return np.concatenate([np.stack([np.cos(2.0 * th), np.sin(2.0 * th)], axis=1),
                           X[:, 2:]], axis=1)


def stability(fa: DenseField, fb: DenseField,
              radius_px: float = 6.0) -> Dict[str, float]:
    """B5: how stable is a dense node's CONTENT across two views, position aside?

    For every node of `fa`, the nearest node of `fb` is found within `radius_px`
    (a POSITIONAL match; no identity is claimed).  Content stability is the
    Pearson correlation, over the matched pairs, of the two views' standardised
    content vectors -- each column z-scored over the pairs so no slot dominates,
    the axial angle carried as cos/sin(2*theta).  Positional stability is the mean
    match distance in px.  Position instability with content stability is the
    EXPECTED result: a dense node is carried by its place, not by its coordinates.
    """
    if not fa.nodes or not fb.nodes:
        return {"n_pairs": 0, "content_r": float("nan"), "mean_dist_px": float("nan")}
    A = content_matrix(fa)
    B = content_matrix(fb)
    posB = np.asarray([(n.x, n.y) for n in fb.nodes], dtype=float)
    pairs: List[Tuple[int, int, float]] = []
    for i, n in enumerate(fa.nodes):
        d = np.hypot(posB[:, 0] - n.x, posB[:, 1] - n.y)
        j = int(np.argmin(d))
        if float(d[j]) <= float(radius_px):
            pairs.append((i, j, float(d[j])))
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "content_r": float("nan"),
                "mean_dist_px": float(np.mean([p[2] for p in pairs]) if pairs else float("nan"))}
    ia = np.asarray([p[0] for p in pairs], dtype=int)
    ib = np.asarray([p[1] for p in pairs], dtype=int)
    Xa = _axial_features(A, ia)
    Xb = _axial_features(B, ib)
    mu_a, sd_a = Xa.mean(axis=0), Xa.std(axis=0)
    mu_b, sd_b = Xb.mean(axis=0), Xb.std(axis=0)
    sd_a = np.where(sd_a < 1e-9, 1.0, sd_a)
    sd_b = np.where(sd_b < 1e-9, 1.0, sd_b)
    Za = ((Xa - mu_a) / sd_a).ravel()
    Zb = ((Xb - mu_b) / sd_b).ravel()
    r = float(np.corrcoef(Za, Zb)[0, 1]) if Za.std() > 1e-12 and Zb.std() > 1e-12 else float("nan")
    return {"n_pairs": len(pairs), "content_r": r,
            "mean_dist_px": float(np.mean([p[2] for p in pairs]))}


def selftest_contrast() -> Dict[str, Any]:
    """B4's own fixture: a uniform patch must produce (almost) nothing, texture must.

    Returns the node counts on four synthetic 128x128 frames so a test can assert
    the field is content-driven rather than a lattice.
    """
    rng = np.random.default_rng(20261118)
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    out: Dict[str, Any] = {}
    uniform = np.full((h, w, 3), 128.0, dtype=np.float32)
    gradient = np.clip(60.0 + 120.0 * (xx / float(w)), 0, 255)[..., None] * np.ones(3)
    noise = np.clip(128.0 + rng.normal(0, 40, (h, w, 3)), 0, 255)
    grating = (128.0 + 60.0 * np.sin(2 * np.pi * xx / 6.0))[..., None] * np.ones(3)
    for name, im in (("uniform", uniform), ("gradient", gradient),
                     ("noise", noise), ("grating", grating)):
        f = dense_field(im.astype(np.uint8))
        out[name] = {"n_nodes": len(f.nodes), "coverage": round(coverage(f, (h, w)), 4)}
    return out


if __name__ == "__main__":  # pragma: no cover - a manual probe
    import json
    import sys
    from PIL import Image
    p = sys.argv[1] if len(sys.argv) > 1 else "data/ETH-80-master/1/1/apple1-035-045.png"
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    f = dense_field(rgb)
    print(json.dumps({"nodes": len(f.nodes), "stats": f.stats,
                      "coverage": coverage(f, rgb.shape[:2])}, indent=1))
