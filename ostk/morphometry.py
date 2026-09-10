"""Level morphometry in the VERTEBRA'S OWN FRAME: body height, end-plate width and
depth, canal width and depth, pedicle width.

WHY THIS MODULE EXISTS. The dimensions a surgeon looks up come from cadaveric series
that read a caliper between two named landmarks on an isolated, cleaned bone.[1][2]
Measuring the same quantity on a segmentation is easy to get almost right and
systematically wrong, in two ways that both inflate and neither of which announces
itself:

  MEASURING ALONG THE SCANNER'S AXES.  A vertebral body height taken as a straight-up
  extent in the volume's frame is not the height of the body; it is that height divided
  by the cosine of however far the body is tilted. Supine L5 sits 20-30 degrees off
  horizontal, which is +6% to +15% on a 24 mm body. The same applies across the pedicle:
  a chord straight across the patient cuts an oblique pedicle diagonally, and lumbar
  pedicle obliquity grows from about 10 degrees at L1 to 30 at L5, so the error grows
  caudally and looks exactly like anatomy.

  TAKING AN EXTREMUM OVER A REGION.  A maximum column height over the posterior half of
  a body is >= a reading at the posterior wall by construction, before any anatomy is
  involved, and it collects every rim osteophyte and every voxel of label bleed into the
  disc space along the way.

Both are fixed the same way: build a frame from the vertebra's own end-plates and take
every dimension between landmarks in it. Height is corner to corner. Width is across the
plate's own left-right. Pedicle width is perpendicular to the pedicle's own long axis,
which is measured rather than assumed.

WHAT THIS IS NOT. It is not tuned to reproduce the published values. An estimator fitted
until it matches a reference cannot then be used to check that reference, and comparing a
measurement against itself is how a cohort's own pelvic incidence ends up in a reference
column. What is matched here is the DEFINITION -- landmark to landmark, in the bone's
frame -- and the numbers then land where they land.

  [1] Panjabi MM, Takata K, Goel V, et al. Thoracic human vertebrae: quantitative
      three-dimensional anatomy. Spine 1991;16(8):888-901.
  [2] Panjabi MM, Goel V, Oxland T, et al. Human lumbar vertebrae: quantitative
      three-dimensional anatomy. Spine 1992;17(3):299-306.

Nomenclature follows those papers so the two are comparable without a translation table:
VBHa/VBHp anterior and posterior body height, EPWu/EPWl and EPDu/EPDl upper and lower
end-plate width and depth, SCW/SCD canal width and depth, PDW pedicle width.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage

from .geometry import WORLD_SUPERIOR, principal_axes, unit  # principal_axes -> (axes3x3, w, mean)
from .masks import binary_mask, largest_component
from .vertebral_body import body_mask, canal_mask, endplate_corners_body

__all__ = [
    "vertebra_frame",
    "body_heights",
    "endplate_dimensions",
    "canal_dimensions",
    "pedicle_widths",
    "level_morphometry",
]


def _world(mask, affine) -> np.ndarray:
    idx = np.array(np.nonzero(np.asarray(mask, bool))).T
    if not len(idx):
        return np.empty((0, 3))
    return (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]


def _span(points: np.ndarray, axis: np.ndarray) -> float:
    """Extent of a point cloud along one direction, in mm."""
    t = points @ unit(axis)
    return float(t.max() - t.min())


def vertebra_frame(sup, inf, *, lr=(1.0, 0.0, 0.0)) -> Optional[Dict[str, np.ndarray]]:
    """A right-handed frame built from this vertebra's OWN two end-plates.

    `sup` and `inf` are (anterior, posterior) world points from
    `endplate_corners_body`. The axial axis is the mean of the two anterior-to-posterior
    chords, so it lies in the plates rather than in the scanner; the superior axis is
    what is left of the world superior once that is projected out, so a tilted body is
    measured along itself; the lateral axis completes the frame.

    Returns None when the two plates are degenerate against each other, which happens on
    a collapsed or badly segmented level and should drop it rather than produce a number.
    """
    sa, sp_ = np.asarray(sup[0], float), np.asarray(sup[1], float)
    ia, ip_ = np.asarray(inf[0], float), np.asarray(inf[1], float)
    ap = unit((sa - sp_) + (ia - ip_))                    # posterior -> anterior
    if not np.isfinite(ap).all():
        return None
    # superior, made orthogonal to the plates' own anteroposterior direction
    s = np.asarray(WORLD_SUPERIOR, float)
    s = s - (s @ ap) * ap
    if np.linalg.norm(s) < 1e-6:
        return None
    s = unit(s)
    lat = np.cross(s, ap)
    if np.linalg.norm(lat) < 1e-6:
        return None
    lat = unit(lat)
    # keep the lateral axis pointing the same way the caller's does, so left and right
    # do not flip between levels
    if lat @ unit(lr) < 0:
        lat = -lat
    return {"ap": ap, "sup": s, "lat": lat}


def body_heights(sup, inf) -> Dict[str, float]:
    """VBHa and VBHp: the two wall heights, corner to corner.

    This is the caliper reading, not an extent: the distance from the superior
    anterior corner to the inferior anterior corner is the anterior wall, whatever
    angle the vertebra sits at in the scanner.
    """
    sa, sp_ = np.asarray(sup[0], float), np.asarray(sup[1], float)
    ia, ip_ = np.asarray(inf[0], float), np.asarray(inf[1], float)
    return {"VBHa": float(np.linalg.norm(sa - ia)),
            "VBHp": float(np.linalg.norm(sp_ - ip_))}


def endplate_dimensions(body, affine, corners, frame, *,
                        plate_band_mm: float = 3.0) -> Dict[str, float]:
    """EPW and EPD for one plate.

    DEPTH is the corner-to-corner chord, which is what a caliper across the plate reads.
    WIDTH is the lateral extent of the plate surface itself, taken in the plate's own
    left-right and over a thin band at the plate rather than over the whole body, because
    a body waists in below the rim and the published width is measured at the rim.
    """
    out: Dict[str, float] = {}
    a, p = np.asarray(corners[0], float), np.asarray(corners[1], float)
    out["EPD"] = float(np.linalg.norm(a - p))
    pts = _world(body, affine)
    if not len(pts):
        return out
    mid = 0.5 * (a + p)
    d = (pts - mid) @ frame["sup"]
    band = pts[np.abs(d) <= plate_band_mm]
    if len(band) >= 10:
        out["EPW"] = _span(band, frame["lat"])
    return out


def canal_dimensions(mask, affine, frame, *, sup_axis=WORLD_SUPERIOR,
                     lr=(1.0, 0.0, 0.0)) -> Dict[str, float]:
    """SCW and SCD, in the vertebra's frame rather than the scanner's.

    ostk's canal_mask carves the enclosed canal from the whole-vertebra mask; both
    dimensions are then extents of that solid along the frame's own axes.
    """
    canal = canal_mask(np.asarray(mask, bool), affine, sup_axis=sup_axis)
    if canal is None or not canal.any():
        return {}
    pts = _world(largest_component(canal), affine)
    if len(pts) < 20:
        return {}
    return {"SCW": _span(pts, frame["lat"]), "SCD": _span(pts, frame["ap"])}


def _fill(m):
    """Fill each axial section. Erosion and distance transforms both eat from interior
    holes, and the basivertebral channel is background in these masks."""
    out = np.zeros_like(m)
    for k in range(m.shape[2]):
        if m[:, :, k].any():
            out[:, :, k] = ndimage.binary_fill_holes(m[:, :, k])
    return out


def _resample_sdf(mask, sp, box, target):
    """Resample a binary mask onto a fine isotropic grid with SUB-VOXEL edges.

    An 8 mm pedicle on a 0.8 mm grid is quantised at 10%, and a staircase edge measured
    across an oblique strut reads systematically wide because the steps protrude
    perpendicular to the direction being measured. Nearest-neighbour zoom only makes the
    staircase finer; a signed distance field carries where the surface really lies between
    voxel centres, so it is interpolated and re-thresholded instead.
    """
    lo, hi = box
    sub = np.asarray(mask, bool)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not sub.any():
        return None
    sdf = (ndimage.distance_transform_edt(sub, sampling=sp)
           - ndimage.distance_transform_edt(~sub, sampling=sp))
    return ndimage.zoom(sdf, np.asarray(sp, float) / target, order=1, mode="nearest") > 0.0


def pedicle_widths(mask, body, affine, frame=None, *, sup_axis=WORLD_SUPERIOR,
                   lr=(1.0, 0.0, 0.0), margin: int = 6,
                   target: float = 0.35) -> Dict[str, float]:
    """PDW per side, by the published convention.

    Written against the literature after four attempts failed. Two things it settles that
    guessing did not:

    THE REGION IS BOUNDED BEFORE ANYTHING IS MINIMISED. Kumar et al. (Int J Med Robot 2025,
    doi:10.1002/rcs.70049) define an Initial Pedicle Region per axial slice, between the
    vertebral-body boundary and the spinal-canal boundary, split left and right by the line
    joining the canal centre to the body centre. The pars is posterior to the canal, so it
    is never a candidate -- which is the whole of my min-cut failure, where the narrowest
    route from body to lamina ran through the pars instead of the pedicle.

    PDW IS THE TRANSVERSE EXTENT, NOT THE INSCRIBED CIRCLE. This is the error that made
    every previous version read flat across levels. A maximum inscribed circle measures
    min(width, height). At L1 the pedicle is 8.6 wide and about 16 tall, so the circle is
    the width and the answer looks right; by L5 it is about 18.6 wide and the height is
    limiting, so the same code returns the height and the caudal widening disappears.
    Makino et al. (Eur Spine J 2012) show the flip directly: transverse 5.5 to 12.9 mm from
    L1 to L5 while sagittal runs 11.9 down to 9.0. So the cross-section is measured along
    the vertebra's own transverse direction.

    Isthmus is the section of least area perpendicular to the pedicle's own axis, after
    Li et al. (Spine 2004;29:2438) and Sugisaki et al. (Spine 2009;34:2599). Outer
    cortical, to match Panjabi and Zindrick.
    """
    # SLOW, AND ABSENT FROM THE 2026-09-10 SHARD RUN -- but not broken. Read this before
    # concluding either way, because I concluded wrongly once already.
    #
    # WHAT IS VERIFIED. Called directly, this works on released volumes: case 0010 returns
    # L1 6.62 mm and L5 15.10 mm, and `level_morphometry` returns the same through its own
    # call path. The L5 figure is nearer the published mean of 16.20 than the release's own
    # extraction code manages (20.70), so the method is the better of the two where it runs.
    #
    # WHAT IS ALSO TRUE. It is EXPENSIVE: the signed-distance resample to 0.35 mm costs
    # roughly 30-60 s per level, so `ostk morph` over two cases exceeded a 280 s timeout.
    # And the 2026-09-10 shard run over all 802 records produced pedicle columns for two
    # level-instances out of ~5,600 -- while completing in 52-90 minutes per shard, which
    # is far less time than 100 cases x 7 levels of this code would take. The columns were
    # not computed and dropped; they were never computed.
    #
    # WHAT IS NOT ESTABLISHED is why. The likeliest reading is that those shards ran an
    # earlier state of this package than the one that now measures correctly. It is NOT
    # the `frame` argument, which this function never reads.
    #
    # So: do not describe ostk as measuring pedicle width in a pipeline until a full run
    # is shown to populate the columns, and budget hours rather than minutes for it.
    vert = np.asarray(mask, bool)
    canal = canal_mask(vert, affine, sup_axis=sup_axis)
    sp = np.abs(np.asarray(affine, float)[:3, :3]).sum(axis=0)
    vfill = _fill(vert)
    idx = np.argwhere(vfill)
    if not len(idx) or canal is None:
        return {}
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, np.array(vfill.shape))
    V = _resample_sdf(vfill, sp, (lo, hi), target)
    B = _resample_sdf(_fill(np.asarray(body, bool)), sp, (lo, hi), target)
    C = _resample_sdf(canal, sp, (lo, hi), target)
    if V is None or C is None or B is None or not C.any():
        return {}
    B = B & V

    # ---- the Initial Pedicle Region, per axial section -------------------------------
    ipr = {"l": np.zeros_like(V), "r": np.zeros_like(V)}
    for k in range(V.shape[2]):
        cs, vs, bs = C[:, :, k], V[:, :, k], B[:, :, k]
        if cs.sum() < 20 or not bs.any():
            continue
        cx = np.nonzero(cs.any(axis=1))[0]
        cy = np.nonzero(cs.any(axis=0))[0]
        bx = np.nonzero(bs.any(axis=1))[0]
        band = np.zeros_like(vs)
        band[:, cy.min():cy.max() + 1] = True          # anterior to the canal's back wall
        for name, lat in (("l", slice(int(bx.min()), int(cx.min()))),
                          ("r", slice(int(cx.max()) + 1, int(bx.max()) + 1))):
            if lat.start is None or lat.stop is None or lat.stop <= lat.start:
                continue
            reg = np.zeros_like(vs)
            reg[lat] = vs[lat]
            reg &= band & ~bs                          # bone, not body, lateral of canal
            if reg.sum() < 15:
                continue
            lab, n = ndimage.label(reg)
            if n:
                sizes = ndimage.sum(reg, lab, range(1, n + 1))
                ipr[name][:, :, k] = lab == (int(np.argmax(sizes)) + 1)

    out = {}
    lat_dir = np.array([1.0, 0.0, 0.0])                # array x is left-right after canon.
    for name in ("l", "r"):
        pts_i = np.argwhere(ipr[name])
        if len(pts_i) < 200:
            continue
        pts = pts_i.astype(float) * target
        axes, _, cen = principal_axes(pts)
        axis = unit(np.asarray(axes)[:, 0])            # the pedicle's own long axis
        t = pts @ axis
        lo_t, hi_t = np.quantile(t, [0.15, 0.85])
        best = None
        for c in np.linspace(lo_t, hi_t, 25):
            sec = pts[np.abs(t - c) <= target]
            if len(sec) < 25:
                continue
            area = len(sec) * target ** 2               # section of least area = isthmus
            if best is None or area < best[0]:
                best = (area, c, sec)
        if best is None:
            continue
        sec = best[2]
        # transverse extent within that section, in the vertebra's own left-right
        u = lat_dir - (lat_dir @ axis) * axis
        if np.linalg.norm(u) < 1e-6:
            continue
        u = unit(u)
        proj = sec @ u
        out[name] = float(proj.max() - proj.min())
    if "l" in out and "r" in out:
        out["mean"] = 0.5 * (out["l"] + out["r"])
    return out


def level_morphometry(label, affine, level_id: int, *, sup_axis=WORLD_SUPERIOR,
                      lr=(1.0, 0.0, 0.0), max_plate_rms_mm: float = 2.0
                      ) -> Optional[Dict[str, float]]:
    """Every dimension above for ONE vertebra, or None if its plates did not fit.

    The plate fit's own residual is the gate. A level whose end-plate did not converge
    has no landmarks, and a landmark measurement without landmarks is a number with
    nothing behind it -- so it is dropped rather than approximated.
    """
    m = binary_mask(np.asarray(label), int(level_id))
    if m.sum() < 200:
        return None
    body = body_mask(m, affine, sup_axis=sup_axis, lr=lr)
    if body is None or body.sum() < 100:
        return None
    sup = endplate_corners_body(m, affine, "superior", sup_axis=sup_axis, lr=lr, body=body)
    inf = endplate_corners_body(m, affine, "inferior", sup_axis=sup_axis, lr=lr, body=body)
    if sup is None or inf is None:
        return None
    if max(float(sup[2]), float(inf[2])) > max_plate_rms_mm:
        return None
    frame = vertebra_frame(sup, inf, lr=lr)
    if frame is None:
        return None

    out: Dict[str, float] = dict(body_heights(sup, inf))
    for which, corners, tag in (("superior", sup, "u"), ("inferior", inf, "l")):
        for k, v in endplate_dimensions(body, affine, corners, frame).items():
            out[f"{k}{tag}"] = v
    out.update(canal_dimensions(m, affine, frame, sup_axis=sup_axis, lr=lr))
    out.update(pedicle_widths(m, body, affine, frame, sup_axis=sup_axis, lr=lr))
    out["plate_rms_mm"] = max(float(sup[2]), float(inf[2]))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
