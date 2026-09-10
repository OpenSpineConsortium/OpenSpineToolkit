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


def pedicle_widths(mask, body, affine, frame, *, min_vox: int = 40,
                   sup_axis=WORLD_SUPERIOR) -> Dict[str, float]:
    """PDW and PDH for each side, at the isthmus, across the pedicle's OWN axis.

    NOT VALIDATED -- opt in with `pedicle=True` and check it before believing it. Against
    Panjabi 1992 this reads roughly half at L5 (about 10 mm against 18.6) while the four
    other dimensions in this module land within about 1.5 mm at every level. Two earlier
    versions failed in opposite directions, which is the signal that the isolation, not
    the measurement, is what is wrong: the pedicle is a short oblique strut continuous
    with the body at one end and the lamina at the other, and separating it from both by
    a rule on the canal's extent is evidently not enough. The dimensions themselves are
    taken correctly once a region is given; the region is the open problem.

    THE PEDICLE HAS TO BE ISOLATED FIRST, and this is where a first version of this
    function went wrong. Taking "everything that is not body" and splitting it at the
    midline hands each side the lamina, the articular processes and the transverse
    process as well, all of which are larger than the pedicle at the upper lumbar
    levels. The principal axis then follows that bulk rather than the strut, and the
    measured waist belongs to the wrong structure: it read 17.0 mm at L1 against a
    published 8.6, and was near-correct at L5 only because an L5 pedicle is big enough
    to dominate its own neighbourhood.

    So the pedicle is localised the way it is defined -- the bridge running lateral to
    the canal, between the body behind it and the lamina behind that. Sections are
    restricted to those where the arch actually encloses a canal, and within them to the
    anterior part of the canal's anteroposterior span, which is where the pedicle
    attaches. That region's own long axis is then measured from its points, and the
    dimensions are taken in the plane across it: the ISTHMUS is a waist, so each is a low
    percentile over the sections rather than an extent of the whole strut, which would be
    measured where it flares into the body.

    PDW is the dimension closer to the frame's lateral axis and PDH the one closer to
    its superior axis, rather than the smaller and larger of the two: at a level where
    the two are close, sorting by size silently swaps their names.
    """
    m = np.asarray(mask, bool)
    b = np.asarray(body, bool) if body is not None else np.zeros_like(m)
    canal = canal_mask(m, affine, sup_axis=sup_axis)
    if canal is None or not canal.any():
        return {}
    arch = m & ~b
    if arch.sum() < 2 * min_vox:
        return {}

    ax = int(np.argmax(np.abs(np.asarray(affine, float)[:3, :3].T @ unit(sup_axis))))
    # WHICH END OF THE CANAL IS ANTERIOR IS NOT AN ASSUMPTION. The pedicle attaches at the
    # end of the canal that faces the body, and which array direction that is depends on
    # the affine. Taking the wrong end measures the lamina instead: it read a flat 7 mm at
    # every level, with none of the caudal widening a pedicle actually has.
    bc = _world(b, affine).mean(axis=0) if b.any() else None
    cc = _world(canal, affine).mean(axis=0)
    apv = frame["ap"] if bc is None else unit(bc - cc)
    # +1 if increasing array index along the in-plane A-P axis moves anteriorly
    inplane = [i for i in range(3) if i != ax]
    A = np.asarray(affine, float)[:3, :3]
    ap_axis_idx = int(np.argmax([abs(A[:, i] @ apv) for i in inplane]))
    ap_arr = inplane[ap_axis_idx]
    ap_sign = 1 if (A[:, ap_arr] @ apv) > 0 else -1
    # position of that array axis within the moved (sup-first) view
    ap_moved = 0 if ap_arr < ax else 1        # after moveaxis(ax, 0), axes are [ax, rest...]
    ap_moved = [i for i in range(3) if i != ax].index(ap_arr)
    keep = np.zeros_like(arch)
    moved_a, moved_c, moved_k = (np.moveaxis(x, ax, 0) for x in (arch, canal, keep))
    for k in range(moved_a.shape[0]):
        ca = moved_c[k]
        if ca.sum() < 12:
            continue
        # in this 2-D section, axis `ap_moved` is anteroposterior and the other is lateral
        lat_moved = 1 - ap_moved
        ys = np.nonzero(ca.any(axis=lat_moved))[0]      # extent along A-P
        xs = np.nonzero(ca.any(axis=ap_moved))[0]       # extent along lateral
        if len(ys) < 3 or len(xs) < 3:
            continue
        span = ys.max() - ys.min()
        if ap_sign > 0:                                  # anterior is the HIGH index end
            lo_ap, hi_ap = ys.min() + int(0.45 * span), ys.max()
        else:                                            # anterior is the LOW index end
            lo_ap, hi_ap = ys.min(), ys.min() + int(0.55 * span)
        band = np.zeros_like(ca)
        lat = np.zeros_like(ca)
        if ap_moved == 1:
            band[:, lo_ap:hi_ap + 1] = True
            lat[:xs.min(), :] = True
            lat[xs.max() + 1:, :] = True
        else:
            band[lo_ap:hi_ap + 1, :] = True
            lat[:, :xs.min()] = True
            lat[:, xs.max() + 1:] = True
        moved_k[k] = moved_a[k] & band & lat
    keep = np.moveaxis(moved_k, 0, ax)
    if keep.sum() < 2 * min_vox:
        return {}

    pts_all = _world(keep, affine)
    if not len(pts_all):
        return {}
    latv, supv = frame["lat"], frame["sup"]
    mid = float(np.median(_world(m, affine) @ latv))
    t = pts_all @ latv
    out: Dict[str, float] = {}
    for side, sel in (("r", t > mid), ("l", t < mid)):
        pts = pts_all[sel]
        if len(pts) < min_vox:
            continue
        try:
            axes, _, _ = principal_axes(pts)
        except Exception:
            continue
        axis = unit(np.asarray(axes)[:, 0])
        # the two directions across the strut
        u = latv - (latv @ axis) * axis
        if np.linalg.norm(u) < 1e-6:
            continue
        u = unit(u)
        v = unit(np.cross(axis, u))
        # name them by which frame axis each is closer to, not by which is smaller
        w_dir, h_dir = (u, v) if abs(u @ latv) >= abs(v @ latv) else (v, u)
        sarr = pts @ axis
        lo, hi = np.quantile(sarr, [0.15, 0.85])
        ws, hs = [], []
        for c in np.linspace(lo, hi, 9):
            sec = pts[np.abs(sarr - c) <= 1.0]
            if len(sec) < 8:
                continue
            ws.append(_span(sec, w_dir))
            hs.append(_span(sec, h_dir))
        if ws:
            out[f"PDW_{side}"] = float(np.percentile(ws, 20))
            out[f"PDH_{side}"] = float(np.percentile(hs, 20))
    for k in ("PDW", "PDH"):
        a, bb = out.get(f"{k}_l"), out.get(f"{k}_r")
        if a is not None and bb is not None:
            out[k] = 0.5 * (a + bb)
    return out


def level_morphometry(label, affine, level_id: int, *, sup_axis=WORLD_SUPERIOR,
                      lr=(1.0, 0.0, 0.0), max_plate_rms_mm: float = 2.0,
                      pedicle: bool = False) -> Optional[Dict[str, float]]:
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
    # PEDICLE IS OFF BY DEFAULT AND THE REASON IS IN pedicle_widths' docstring: it is not
    # validated. Every other dimension here lands within about 1.5 mm of the published
    # cadaveric series across levels; this one does not, and shipping a number that cannot
    # be checked next to four that can would borrow their credibility.
    if pedicle:
        out.update(pedicle_widths(m, body, affine, frame, sup_axis=sup_axis))
    out["plate_rms_mm"] = max(float(sup[2]), float(inf[2]))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
