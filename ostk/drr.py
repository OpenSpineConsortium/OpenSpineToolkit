"""ostk.drr -- sagittal-plane DRR rendering along the TRUE patient
bicoxofemoral axis, with correct isotropic pixel spacing.

This exists because the (separately maintained, not a dependency of this
repo) XRSpinoPelvic1K project's DRR renderer approximates "lateral" as
whichever CT voxel axis is nearest to the world L-R direction, and never
resamples for the source CT's (generally anisotropic) voxel spacing. Both
were confirmed empirically: on case 0033, that axis choice differed from the
true bicoxofemoral axis by ~23 degrees (the two femoral heads, which should
coincide under a true lateral projection, showed ~80mm apart after unit
conversion), and voxel v/h spacing mismatch across cases ranged 2.4%-46.8%
with zero resampling to correct it.

This module reuses ostk.project2d's own axis/basis (the same `lr` and
`sagittal_axes` used for the PI/SS/PT/LL landmark projection) as the
projection geometry, so a render from this module and project2d's landmark
output share an exact origin/basis -- not an approximation of one. Volume
integration along `lr` is via trilinear interpolation
(scipy.ndimage.map_coordinates), since `lr` is essentially never aligned with
the CT's own voxel grid for a real patient. The HU->attenuation intensity
formula is the same simple bone-emphasis idea as XRSpinoPelvic1K's (not
imported from it -- no dependency taken), just re-expressed with an
integration grid this module controls.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage

from .geometry import WORLD_SUPERIOR, fit_sphere, project_to_plane_2d, unit
from .labels import lid
from .masks import binary_mask, endplate_points, largest_component, mask_world, surface_slab
from .metrics import LL_ENDPLATE_CHAIN
from .project2d import sagittal_axes

DRR_METHOD_VERSION = "ostk-drr-v1"

DEFAULT_MARGIN_MM = 30.0
MIN_FOV_MM = (260.0, 420.0)          # (width, height) floor, in mm
FEMUR_CRANIAL_CAP_MM = 60.0          # cap femur's contribution to framing at
                                     # head-center +/- this, for consistent
                                     # framing regardless of how much shaft a
                                     # given scan's FOV happens to include


def _pi_axis_and_origin(label, affine, sup_axis=WORLD_SUPERIOR,
                        endplate_frac: float = 0.15, head_frac: float = 0.35,
                        min_voxels: int = 50) -> Optional[Dict]:
    """Mirrors ostk.project2d.pelvic_incidence_2d_from_label's own extraction
    (same primitives/fractions/QC gate) -- returns the raw femoral-head
    centers, bicox origin, and lr axis this module projects along. None if a
    landmark is unavailable (never silently renders a wrong axis)."""
    s1 = binary_mask(label, lid("S1"))
    src = s1 if s1.any() else binary_mask(label, lid("sacrum"))
    ep = endplate_points(largest_component(src), affine, sup_axis, "superior", endplate_frac)
    fl = mask_world(largest_component(binary_mask(label, lid("femur_left"))), affine)
    fr = mask_world(largest_component(binary_mask(label, lid("femur_right"))), affine)
    fhl = surface_slab(fl, sup_axis, "superior", head_frac)
    fhr = surface_slab(fr, sup_axis, "superior", head_frac)
    if min(len(ep), len(fhl), len(fhr)) < min_voxels:
        return None
    cL, _, _ = fit_sphere(fhl)
    cR, _, _ = fit_sphere(fhr)
    bicox = 0.5 * (cL + cR)
    return {"cL": cL, "cR": cR, "bicox": bicox, "lr": unit(cR - cL)}


def _framing_points(label, affine, sup_axis=WORLD_SUPERIOR,
                    head_frac: float = 0.35) -> np.ndarray:
    """World-mm points (S1/sacrum + cranially-capped femurs + present lumbar
    levels) used only to size/center the render's field of view -- kept
    separate from the PI axis extraction above since framing wants the whole
    labeled silhouette, not just the fitted landmark points."""
    pts = []

    s1 = binary_mask(label, lid("S1"))
    src = s1 if s1.any() else binary_mask(label, lid("sacrum"))
    w = mask_world(largest_component(src), affine)
    if len(w):
        pts.append(w)

    for fem in ("femur_left", "femur_right"):
        fw = mask_world(largest_component(binary_mask(label, lid(fem))), affine)
        if not len(fw):
            continue
        head = surface_slab(fw, sup_axis, "superior", head_frac)
        if not len(head):
            continue
        c, _, _ = fit_sphere(head)
        d = (fw - c) @ unit(sup_axis)
        capped = fw[np.abs(d) <= FEMUR_CRANIAL_CAP_MM]
        if len(capped):
            pts.append(capped)

    for lv in LL_ENDPLATE_CHAIN:
        if lv == "S1":
            continue
        m = binary_mask(label, lid(lv))
        if m.any():
            pts.append(mask_world(largest_component(m), affine))

    return np.concatenate(pts, axis=0)


def sagittal_drr_from_label(label, ct_volume, affine, *, sup_axis=WORLD_SUPERIOR,
                            pixel_spacing_mm: float = 1.0,
                            margin_mm: float = DEFAULT_MARGIN_MM,
                            min_fov_mm=MIN_FOV_MM, emphasis: str = "bone",
                            bone_hu: float = 150.0, hu_floor: float = -1000.0,
                            hu_ceil: float = 2000.0, gamma: float = 1.0,
                            endplate_frac: float = 0.15, head_frac: float = 0.35,
                            min_voxels: int = 50, interp_order: int = 1
                            ) -> Optional[Dict]:
    """Render a lateral-equivalent sagittal DRR integrated along the patient's
    TRUE bicoxofemoral axis (ostk.project2d's own `lr`/`sagittal_axes`), at a
    single isotropic pixel spacing chosen by the caller -- not the nearest CT
    voxel axis, and not the CT's own (generally non-isotropic) voxel spacing.

    Field of view is sized from the labeled S1/sacrum + femur + lumbar
    silhouette (see _framing_points), not the fitted landmark points, so the
    frame reads as a complete lateral view rather than a crop around a few
    points; `min_fov_mm` floors it for sparse-label cases.

    Intensity mapping mirrors XRSpinoPelvic1K's bone-emphasis idea (mu =
    ReLU(HU - bone_hu) for 'bone', else a windowed HU) -- reimplemented here,
    not imported, since this module takes no dependency on that project.

    Returns None if the PI landmarks aren't extractable (mirrors
    project2d/metrics' QC gate -- never silently renders along a wrong axis).
    """
    axis_info = _pi_axis_and_origin(label, affine, sup_axis, endplate_frac,
                                    head_frac, min_voxels)
    if axis_info is None:
        return None
    lr, origin = axis_info["lr"], axis_info["bicox"]
    ant, cranial = sagittal_axes(lr, sup_axis)

    framing_pts = _framing_points(label, affine, sup_axis, head_frac)
    uv = project_to_plane_2d(framing_pts, origin, ant, cranial)
    u_min, v_min = uv.min(axis=0) - margin_mm
    u_max, v_max = uv.max(axis=0) + margin_mm
    depth = (framing_pts - origin) @ lr
    d_min, d_max = float(depth.min()) - margin_mm, float(depth.max()) + margin_mm

    width_mm = max(u_max - u_min, min_fov_mm[0])
    height_mm = max(v_max - v_min, min_fov_mm[1])
    u_c, v_c = 0.5 * (u_min + u_max), 0.5 * (v_min + v_max)
    u_min, u_max = u_c - width_mm / 2, u_c + width_mm / 2
    v_min, v_max = v_c - height_mm / 2, v_c + height_mm / 2

    W = max(int(round(width_mm / pixel_spacing_mm)), 1)
    H = max(int(round(height_mm / pixel_spacing_mm)), 1)
    us = u_min + (np.arange(W) + 0.5) * pixel_spacing_mm
    vs = v_min + (np.arange(H) + 0.5) * pixel_spacing_mm
    U, V = np.meshgrid(us, vs)                          # each (H, W)

    n_depth = max(int(round((d_max - d_min) / pixel_spacing_mm)), 1)
    depths = d_min + (np.arange(n_depth) + 0.5) * pixel_spacing_mm

    inv_affine = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    vol = np.asarray(ct_volume, dtype=np.float32)
    cval = hu_floor
    accum = np.zeros((H, W), dtype=np.float64)
    base = origin[None, None, :] + U[..., None] * ant + V[..., None] * cranial
    for d in depths:
        world = base + d * lr                           # (H, W, 3)
        flat = world.reshape(-1, 3)
        homog = np.c_[flat, np.ones(len(flat))]
        ijk = (homog @ inv_affine.T)[:, :3]
        hu = ndimage.map_coordinates(vol, ijk.T, order=interp_order,
                                     mode="constant", cval=cval).reshape(H, W)
        if emphasis == "bone":
            mu = np.clip(hu - bone_hu, 0.0, hu_ceil - bone_hu)
        else:
            mu = np.clip(hu, hu_floor, hu_ceil) - hu_floor
        accum += mu

    img = accum.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    if gamma != 1.0:
        img = img ** float(gamma)
    img = img[::-1, ::-1]                # superior up, anterior left (matches xrsp's display convention)

    return {
        "image": img,
        "pixel_spacing_mm": float(pixel_spacing_mm),
        "origin_world_mm": origin.tolist(),
        "axes": {"anterior": ant.tolist(), "cranial": cranial.tolist(), "lr": lr.tolist()},
        "shape": [H, W],
        "fov_mm": [float(width_mm), float(height_mm)],
        "method_version": DRR_METHOD_VERSION,
    }
