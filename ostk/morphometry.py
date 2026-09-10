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

from .geometry import WORLD_SUPERIOR, principal_axes, unit
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


def pedicle_widths(mask, body, affine, frame, *, min_vox: int = 40) -> Dict[str, float]:
    """PDW for each side, measured ACROSS THE PEDICLE'S OWN AXIS.

    The pedicle is a short oblique strut, and lumbar obliquity runs from roughly 10
    degrees at L1 to 30 at L5. A width taken straight across the patient therefore cuts
    it on the diagonal and reads wide, and reads progressively wider the further caudal
    it goes -- a bias with the shape of an anatomical trend.

    So: take everything that is neither body nor canal, split it at the midline, keep
    each side's largest component, measure its long axis, and take the width in the plane
    across that axis. The isthmus is the WAIST of the strut, so the reported width is a
    low percentile of the per-section widths rather than the extent of the whole strut,
    which would be measured where it flares into the body.
    """
    m = np.asarray(mask, bool)
    b = np.asarray(body, bool) if body is not None else np.zeros_like(m)
    arch = m & ~b
    if arch.sum() < 2 * min_vox:
        return {}
    pts_all = _world(arch, affine)
    if not len(pts_all):
        return {}
    lat = frame["lat"]
    t = pts_all @ lat
    mid = float(np.median(_world(m, affine) @ lat))
    out: Dict[str, float] = {}
    for side, sel in (("r", t > mid), ("l", t < mid)):
        pts = pts_all[sel]
        if len(pts) < min_vox:
            continue
        # the strut's own long axis, from the side's point cloud
        try:
            e1, _, _ = principal_axes(pts)
        except Exception:
            continue
        axis = unit(e1)
        # two directions across it, both perpendicular to the long axis
        u = frame["lat"] - (frame["lat"] @ axis) * axis
        if np.linalg.norm(u) < 1e-6:
            continue
        u = unit(u)
        v = unit(np.cross(axis, u))
        s = pts @ axis
        lo, hi = np.quantile(s, [0.15, 0.85])          # ignore both flares
        widths = []
        for c in np.linspace(lo, hi, 9):
            sec = pts[np.abs(s - c) <= 1.0]
            if len(sec) < 8:
                continue
            widths.append(min(_span(sec, u), _span(sec, v)))
        if widths:
            out[f"PDW_{side}"] = float(np.percentile(widths, 20))
    if "PDW_l" in out and "PDW_r" in out:
        out["PDW"] = 0.5 * (out["PDW_l"] + out["PDW_r"])
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
    out.update(pedicle_widths(m, body, affine, frame))
    out["plate_rms_mm"] = max(float(sup[2]), float(inf[2]))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
