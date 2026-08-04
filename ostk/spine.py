"""ostk.spine — vertebral-body endplate fitting (a reusable primitive).

The superior/inferior endplate is the disc-bearing surface of the vertebral
BODY. Two facts make a naive fit wrong:

  * Posterior elements (canal, facets, spinous/transverse processes) are one
    connected component with the body in a 3-D mask, so they can't be split off
    by connectivity — they must be dropped by ANTERIOR position.
  * The endplate is tilted (sacral slope, wedging), so a flat "top-N% by height"
    slab under-reads the tilt. The true face is the extreme voxel per in-plane
    column along the cranio-caudal axis.

`fit_endplate` handles both and returns a plane (centroid, cranial unit normal,
rms). It's used by `ostk.metrics` (lumbar lordosis) and the demo exporter, and is
the place to improve endplate fitting for the whole toolbox.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, fit_plane_tls, unit


def anterior_axis(normal_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Unit anterior axis: in the sagittal plane (⊥ L–R and ⊥ cranial), oriented
    to world +Y (RAS anterior)."""
    ap = unit(np.cross(np.asarray(lr, float), unit(normal_axis)))
    return ap if ap @ np.array([0.0, 1.0, 0.0]) >= 0 else -ap


def endplate_surface(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                     ap_band=(0.3, 0.9), lat_frac: float = 0.55, nbins: int = 22,
                     lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """The endplate face of a body point cloud (N,3 world mm): keep the central
    `lat_frac` in L–R (drops the lateral sacral alae / transverse processes) and a
    central ANTERIOR band `ap_band` (quantiles along the anterior axis) — the lower
    bound drops posterior elements, the UPPER bound drops the anterior osteophyte
    lip that otherwise tilts the fit (the L1 failure mode). Then take the extreme
    voxel per in-plane column along `normal_axis` (topmost for 'superior')."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return P
    a = unit(normal_axis)
    lrv = unit(lr)
    if 0.0 < lat_frac < 1.0:                            # central medial band only
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
        if len(P) == 0:
            return P
    ap_lo, ap_hi = ap_band
    if 0.0 <= ap_lo < ap_hi <= 1.0 and (ap_lo > 0.0 or ap_hi < 1.0):
        proj = (P - P.mean(0)) @ anterior_axis(a, lr)
        lo, hi = np.quantile(proj, [ap_lo, ap_hi])
        P = P[(proj >= lo) & (proj <= hi)]
        if len(P) == 0:
            return P
    ref = np.array([1.0, 0, 0]) if abs(a @ np.array([1.0, 0, 0])) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(ref - (ref @ a) * a)
    e2 = np.cross(a, e1)
    u, v, w = P @ e1, P @ e2, P @ a
    ui = np.floor((u - u.min()) / (np.ptp(u) + 1e-9) * nbins).astype(int)
    vi = np.floor((v - v.min()) / (np.ptp(v) + 1e-9) * nbins).astype(int)
    key = ui * (nbins + 1) + vi
    sgn = -1.0 if which == "superior" else 1.0          # superior -> max w first
    order = np.lexsort((sgn * w, key))
    sk = key[order]
    first = np.ones(len(order), bool)
    first[1:] = sk[1:] != sk[:-1]
    return P[order[first]]


def endplate_corners(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                     lat_frac: float = 0.70, drop_post: float = 0.30,
                     ant_skip: float = 0.08, corner_win: float = 0.15,
                     reject_k: float = 1.8, nbins: int = 26, lr=(1.0, 0.0, 0.0)):
    """The two cortical CORNERS that define the clinical AP-corner + tangent endplate
    line (methods 1+3), found body-first and off-plate-robustly:

      1. medial band (drops lateral processes / sacral alae);
      2. top SURFACE: the disc-facing cortical voxel per (A-P, L-R) cell — a true 2-D
         surface, so deviations along the plate NORMAL can be judged;
      3. BODY: drop the posterior `drop_post` of the A-P extent (pedicle / canal /
         spinous process / dorsal sacrum) BY POSITION — not by height, which would
         chop the low anterior corner of a tilted sacral endplate;
      4. PCA plate fit, iteratively REJECTING points that deviate along the plate
         normal (the small PCA axis, ≈ S-I): the superior articular facet juts ABOVE
         the plate and the sacral canal / nerve-root hollow plunges BELOW it — both
         are off the endplate, while the A-P–L-R plate itself is not;
      5. corners: posterior corner at the back of the cleaned surface; anterior corner
         a little INSIDE the margin (`ant_skip`) — the tangent variant — so an
         anterior osteophyte lip / edge artifact is bridged, not chased.

    Returns (anterior_corner, posterior_corner, surface) or None. The chord through
    the corners bridges endplate concavity and works on both concave (lumbar) and
    convex (sacral promontory) endplates."""
    P = np.asarray(points, dtype=np.float64)
    a = unit(normal_axis)
    lrv = unit(lr)
    ap = anterior_axis(a, lr)
    if 0.0 < lat_frac < 1.0:
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
    if len(P) < 6:
        return None
    sgn = 1.0 if which == "superior" else -1.0          # which cortical face
    # top SURFACE: the disc-facing cortical voxel per (A-P, L-R) cell
    sc = (P - P.mean(0)) @ ap
    lc = (P - P.mean(0)) @ lrv
    si = np.floor((sc - sc.min()) / (np.ptp(sc) + 1e-9) * nbins).astype(int)
    li = np.floor((lc - lc.min()) / (np.ptp(lc) + 1e-9) * nbins).astype(int)
    key = si * (nbins + 1) + li
    order = np.lexsort((-sgn * (P @ a), key))           # disc-facing voxel first per cell
    ks = key[order]
    first = np.ones(len(order), bool)
    first[1:] = ks[1:] != ks[:-1]
    surf = P[order[first]]
    if len(surf) < 6:
        return None
    # BODY: drop posterior elements BY A-P POSITION (not height)
    pr = (surf - surf.mean(0)) @ ap
    pr = (pr - pr.min()) / (np.ptp(pr) + 1e-9)
    body = surf[pr >= drop_post]
    if len(body) < 6:
        body = surf
    # robust PCA plate; reject points that deviate along the plate NORMAL (≈ S-I):
    # the articular facet (above) and the canal / nerve-root hollow (below).
    keep = np.ones(len(body), bool)
    for _ in range(10):
        c, n, _ = fit_plane_tls(body[keep])
        r = (body - c) @ n
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med)) + 1e-6
        # floor the threshold so we don't reject mere voxel-discretization noise on a
        # near-flat plate (a real facet / canal deviates by ~cm, far above the floor)
        nk = np.abs(r - med) <= max(reject_k * mad, 1.5)
        if int(nk.sum()) == int(keep.sum()) or nk.sum() < 6:
            break
        keep = nk
    body = body[keep]
    # tangent corners on the cleaned endplate surface
    bpr = (body - body.mean(0)) @ ap
    bpr = (bpr - bpr.min()) / (np.ptp(bpr) + 1e-9)      # 0 = posterior, 1 = anterior
    post = body[bpr <= corner_win]
    ant = body[(bpr >= 1 - ant_skip - corner_win) & (bpr <= 1 - ant_skip)]
    if len(post) == 0:
        post = body[[int(np.argmin(bpr))]]
    if len(ant) == 0:
        ant = body[[int(np.argmax(bpr))]]
    Pc = post[np.argmax(sgn * (post @ a))]
    A = ant[np.argmax(sgn * (ant @ a))]
    return A, Pc, body


def endplate_corners_anatomic(points, normal_axis=WORLD_SUPERIOR,
                              which: str = "superior", lr=(1.0, 0.0, 0.0), *,
                              rim_mm: float = 6.0, ant_pct: float = 99.0,
                              lat_frac_rim: float = 0.30, **fit_kw):
    """`endplate_corners`' line with its ANTERIOR end run out to the cortical margin.

    `endplate_corners` is tuned to protect the ANGLE: `ant_skip=0.08` sets the anterior
    corner 8% inside the margin so an osteophyte lip is bridged rather than chased. That
    is right for a Cobb line and wrong for an annotation -- measured on case 0003 the
    anterior corner lands 6-14 mm inside the true anterior cortex, a quarter to a third
    of the body depth, which is plainly visible on a rendered overlay.

    Only the anterior end is touched. `drop_post` was suspected too, but the A-P
    occupancy of the medial band shows it already lands the posterior corner at the
    spinal canal on every level, which is exactly where the posterior corner belongs:

        L3  .##.#.##.##.##.#.##.##.##.##.#.##.##.#.....#.##.##.##.#.##...
                                                 P            (canal = '.....')

    Two earlier approaches failed and are worth not repeating. Both walked the fitted
    line looking for where it left bone:
      * on a DRR -- no gap exists, because the pedicles superimpose over the canal;
      * in 3-D along the line -- the line is a CHORD across a concave endplate, by
        design (it bridges the concavity the way a radiologist draws it), so it does
        not lie on bone at all. L1's superior chord is over air for most of its length:
            .#.##.#.##....#.............#####
        Occupancy sampled along the chord is therefore meaningless, and reading it as
        anatomy put L3's posterior corner past the canal and into the lamina.
    The corner is an A-P EXTENT of the body, not a feature along the chord.

    So: take medial-band points lying within `rim_mm` of the fitted plate (i.e. the
    endplate rim, not the whole body -- the anterior cortex is concave in the sagittal
    midline, so the anterior-most bone sits at the rims and the two faces must not be
    conflated), take their `ant_pct` percentile along the anterior axis so a single
    osteophyte spike cannot win, and slide the anterior corner ALONG THE EXISTING LINE
    until it reaches that A-P coordinate.

    Sliding along the line is what makes this safe: the corner stays collinear with the
    fitted plate, so the line's DIRECTION -- hence SS, LL, PI and PT -- is unchanged.
    Verified on 0003: worst direction change 1e-6 deg, anterior shortfall 6-14 mm -> ~0.

    Returns (anterior_corner, posterior_corner, surface) or None -- same shape as
    `endplate_corners`, so it is a drop-in.
    """
    P = np.asarray(points, dtype=np.float64)
    res = endplate_corners(P, normal_axis, which, lr=lr, **fit_kw)
    if res is None:
        return None
    A, Pc, surf = np.asarray(res[0], float), np.asarray(res[1], float), res[2]
    u = Pc - A
    span = float(np.linalg.norm(u))
    if span <= 0:
        return res
    u = u / span                                     # anterior -> posterior
    lrv = unit(lr)
    ap = anterior_axis(unit(normal_axis), lr)
    denom = float(u @ ap)
    if abs(denom) < 1e-6:                            # line perpendicular to A-P: nothing to slide
        return res
    plate_n = unit(np.cross(u, lrv))                 # in-sagittal, perpendicular to the chord

    if 0.0 < lat_frac_rim < 1.0:                     # medial band
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac_rim) / 2, 1 - (1 - lat_frac_rim) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
    rel = P - A
    rim = P[np.abs(rel @ plate_n) <= rim_mm]         # points on the endplate RIM
    if len(rim) < 6:
        return res
    target = float(np.percentile(rim @ ap, ant_pct))
    t = (target - float(A @ ap)) / denom
    if not np.isfinite(t) or t >= 0.0:               # never pull the corner posteriorly
        return res
    return (A + t * u), Pc, surf


def corner_params_for_level(level: str) -> dict:
    """Body-isolation params. TIGHT isolation (exclude the lateral & posterior
    structure) for every level: on a vertebra it keeps the endplate line off the
    superior articular facet, and on the sacrum it fits the flatter anterior
    endplate rather than riding up the promontory. Kept as a hook for per-level
    overrides, but currently uniform."""
    return dict(lat_frac=0.45, drop_post=0.42)


def fit_endplate(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                 method: str = "corner", ap_band=(0.3, 0.9), lat_frac: float = 0.70,
                 drop_post: float = 0.30, lr=(1.0, 0.0, 0.0), min_points: int = 30
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Fit the superior/inferior endplate plane of a vertebral-body point cloud.
    Returns (centroid, unit normal oriented cranially for 'superior', rms) or None.

    `method='corner'` (default) is the clinical AP-corner + tangent method: the
    endplate line runs through the anterior- and posterior-superior cortical
    corners, BRIDGING endplate concavity (standard anatomy) the way a radiologist
    draws a Cobb line. `method='surface'` is the biomechanical best-fit to the
    cortical top-surface (least-squares) — truer to the whole surface area but
    pulled into the concavity, so it is not used for sagittal-alignment angles."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < min_points:
        return None
    if method == "corner":
        res = endplate_corners(P, normal_axis, which, lat_frac=lat_frac,
                               drop_post=drop_post, lr=lr)
        if res is None:
            return None
        A, Pc, body = res
        mid = 0.5 * (A + Pc)
        n = unit(np.cross(unit(lr), unit(Pc - A)))
        rms = float(np.sqrt(np.mean(((body - mid) @ n) ** 2)))
        a = unit(normal_axis)
        if (which == "superior") != (n @ a >= 0):
            n = -n
        return mid, n, rms
    surf = endplate_surface(P, normal_axis, which, ap_band, lat_frac, lr=lr)
    if len(surf) < min_points:
        surf = P
    c, n, rms = fit_plane_tls(surf)
    # Iteratively reject outliers (MAD-based) so the plane converges to the
    # dominant FLAT endplate: discards anterior osteophyte lips (high outliers)
    # and the posterior down-slope toward the canal/ala (low outliers).
    for _ in range(6):
        d = np.abs((surf - c) @ n)
        thr = 2.0 * np.median(d) + 1e-6
        keep = d <= thr
        if keep.all() or keep.sum() < min_points:
            break
        surf = surf[keep]
        c, n, rms = fit_plane_tls(surf)
    a = unit(normal_axis)
    cranial = n @ a >= 0
    if (which == "superior") != cranial:
        n = -n
    return c, n, rms


def endplate_overmask_midpoint(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                               lr=(1.0, 0.0, 0.0), lo_pct: float = 3.0,
                               hi_pct: float = 97.0, **corner_kw):
    """Midpoint of the endplate portion that is OVER the body mask, kept ON the rim.

    The cleaned surface points are the per-column tops of the body, so their extent
    along the endplate direction is the over-mask span; take its (outlier-robust)
    centre and PROJECT it onto the anterior–posterior corner (rim) line, so the point
    sits on the cortical endplate rather than dipping into the endplate concavity.
    Use this as the PI/PT radius origin / construction anchor. It does NOT change the
    endplate orientation (the rim line / normal), so lordosis is unaffected.
    Returns a world-mm point, or None."""
    res = endplate_corners(points, normal_axis, which, lr=lr, **corner_kw)
    if res is None:
        return None
    A, Pc, body = res
    e_dir = unit(np.cross(unit(lr), unit(Pc - A)))
    c0 = body.mean(axis=0)
    proj = (body - c0) @ e_dir
    lo, hi = np.percentile(proj, [lo_pct, hi_pct])
    center = c0 + 0.5 * (lo + hi) * e_dir          # over-mask A-P centre
    el = unit(Pc - A)                              # rim line direction
    return A + float((center - A) @ el) * el       # on the rim, at the over-mask centre


def endplate_overmask_midpoint_from_label(label, affine, level: str,
                                          normal_axis=WORLD_SUPERIOR,
                                          which: str = "superior", lr=(1.0, 0.0, 0.0)):
    """`endplate_overmask_midpoint` straight from a label volume + structure name
    (S1 falls back to the sacrum label)."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world
    m = binary_mask(label, lid(level))
    if level == "S1" and not m.any():
        m = binary_mask(label, lid("sacrum"))
    pts = mask_world(largest_component(m), affine)
    return endplate_overmask_midpoint(pts, normal_axis, which, lr=lr,
                                      **corner_params_for_level(level))


def endplate_from_label(label, affine, level: str, which: str = "superior",
                        normal_axis=WORLD_SUPERIOR, method: str = "corner",
                        ap_band=(0.3, 0.9), lr=(1.0, 0.0, 0.0), min_points: int = 30):
    """Convenience: fit an endplate straight from a label volume + structure name,
    with body-isolation params chosen for the level (tight for vertebrae, loose for
    the sacrum). For S1 falls back to the sacrum label if the carved S1 is absent."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world
    m = binary_mask(label, lid(level))
    if level == "S1" and not m.any():
        m = binary_mask(label, lid("sacrum"))
    pts = mask_world(largest_component(m), affine)
    return fit_endplate(pts, normal_axis, which, method=method, ap_band=ap_band,
                        lr=lr, min_points=min_points, **corner_params_for_level(level))


# ── the single source of truth for the PI/PT anchor ──────────────────────────────

PI_ANCHOR_DEFAULT = "corner"


def pi_anchor_point(label, affine, *, sup_axis=WORLD_SUPERIOR, mode=PI_ANCHOR_DEFAULT,
                    level: str = "S1", which: str = "superior"):
    """The point PI and PT are measured FROM on the S1 superior endplate.

    ONE definition, used by every caller. It previously lived in three places --
    metrics._pi_from_label_core and two sites in surgery (compensate_pelvis and the
    pelvic-anteversion sign) -- which is how the anchor silently diverged: the surgery
    helpers drove a rotation using one anchor while spinopelvic_summary_from_label
    scored the result with another, so compensate_pelvis undershot its PT target.

    mode="corner"    bisect the endplate between its anterior and posterior corners.
        The operational method PI was DEFINED with on lateral radiographs
        (Legaye/Duval-Beaupere), so it carries the convention the published PI norms
        and Schwab targets were calibrated against -- and it is the only anchor
        derivable from landmarks visible on a radiograph, which keeps CT- and
        XR-derived PI on one definition.
    mode="overmask"  centre of the endplate portion backed by vertebral body,
        projected onto the rim. Retained as a primitive; no longer the default
        anywhere. Interbody cage seating does NOT use it -- that path goes through
        endplate_corners.

    Both lie on the same rim line, so NEITHER changes SS or LL; only PI and PT move
    (2.0 deg on case 0003, where they sit 7.4 mm apart along the rim).

    Returns a world-mm point, or None if the endplate is unavailable.
    """
    from .masks import binary_mask, largest_component, mask_world
    from .labels import LABELS
    lid = LABELS.get(level)
    if lid is None:
        return None
    if mode == "overmask":
        return endplate_overmask_midpoint_from_label(label, affine, level, sup_axis, which)
    if mode != "corner":
        raise ValueError(f"pi anchor mode must be 'corner' or 'overmask', got {mode!r}")
    try:
        pts = mask_world(largest_component(binary_mask(np.asarray(label), lid)), affine)
        kw = {k: v for k, v in corner_params_for_level(level).items()
              if k in endplate_corners.__code__.co_varnames}
        c = endplate_corners(pts, normal_axis=sup_axis, which=which, **kw)
    except Exception:                                        # noqa: BLE001
        return None
    if c is None:
        return None
    return 0.5 * (np.asarray(c[0], float) + np.asarray(c[1], float))
